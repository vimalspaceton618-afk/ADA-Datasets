"""ADA Provenance Lattice v0.1: conservative, auditable JSONL curation.

Design: approved provenance -> text integrity -> PII exclusion -> hygiene/quality
-> independent benchmark decontamination -> exact/fuzzy dedup -> stable splits
-> reproducible manifest. All content remains local, with no automatic crawling.

This is a bounded single-host implementation; neither a multilingual language-ID
model nor a production-scale distributed data curation service.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from dataclasses import asdict, dataclass
from hashlib import sha256
import html
import json
import os
from pathlib import Path
import random
import re
import shutil
import sqlite3
import tempfile
import unicodedata
from typing import BinaryIO, Iterator
from urllib.parse import urlsplit, urlunsplit

from .language import script_consistent
from .lid import accepted_prediction

VERSION = "0.4.0"
WORDS = re.compile(r"\w+", re.UNICODE)
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){10,15}(?!\d)")
PRIVATE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{24,}", re.I)
CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
SPLIT_BUCKETS = {"train": 9800, "validation": 100, "test": 100}
MINHASH_PRIME = (1 << 61) - 1


@dataclass(frozen=True)
class CurationConfig:
    seed: int = 2026
    min_chars: int = 100
    min_words: int = 18
    min_quality: float = 0.65
    max_chars: int = 1_000_000
    min_near_jaccard: float = 0.85
    max_holdout_records: int = 20_000
    supported_languages: tuple[str, ...] = ("en", "ta", "hi")
    script_sanity: bool = False
    lid_min_confidence: float = 0.80
    commercial_use: bool = False
    as_of_date: str | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (self.min_chars < 1 or self.min_words < 1 or
                self.max_chars < self.min_chars or
                not 0 <= self.min_quality <= 1 or
                not 0.5 <= self.min_near_jaccard <= 1 or
                self.max_holdout_records < 1 or
                type(self.script_sanity) is not bool or
                not 0 < self.lid_min_confidence <= 1 or
                type(self.commercial_use) is not bool):
            raise ValueError("invalid curation thresholds")
        if self.as_of_date is not None:
            if not isinstance(self.as_of_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.as_of_date):
                raise ValueError("as_of_date must be YYYY-MM-DD")
            date.fromisoformat(self.as_of_date)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def file_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_hashed_lines(
    stream: BinaryIO, max_bytes: int, *, input_hasher=None,
) -> Iterator[tuple[bytes | None, str]]:
    """Yield bounded physical JSONL records and their byte-exact SHA-256.

    An oversized line is read incrementally and represented by None. Its digest
    still includes every byte, including the line ending when present.
    """
    while chunk := stream.readline(max_bytes + 1):
        if input_hasher is not None:
            input_hasher.update(chunk)
        if len(chunk) <= max_bytes:
            yield chunk, digest(chunk)
            continue
        line_hasher = sha256(chunk)
        while not chunk.endswith(b"\n"):
            chunk = stream.readline(1024 * 1024)
            if not chunk:
                break
            line_hasher.update(chunk)
            if input_hasher is not None:
                input_hasher.update(chunk)
        yield None, line_hasher.hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("duplicate JSON key")
        record[key] = value
    return record


def _invalid_json_constant(_: str):
    raise ValueError("nonfinite JSON value")


def strict_json(data: str) -> object:
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_json_constant)


def canonical_text(text: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    return "\n".join(" ".join(line.split()) for line in text.splitlines()).strip()


def fingerprint(text: str) -> str:
    return digest(text.encode("utf-8"))


def clean_url(value: object) -> str | None:
    """Keep origin and path; discard query, fragments, and embedded credentials."""
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError("invalid_url")
    parts = urlsplit(value)
    if (parts.scheme.lower() not in ("https", "http") or not parts.hostname or
            parts.username or parts.password):
        raise ValueError("invalid_url")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))


def _luhn_card(candidate: str) -> bool:
    digits = "".join(c for c in candidate if c.isdigit())
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for pos, char in enumerate(reversed(digits)):
        value = int(char)
        if pos % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def has_pii_or_secret(text: str) -> bool:
    if EMAIL.search(text) or PRIVATE.search(text):
        return True
    if any(_luhn_card(match.group()) for match in CARD_CANDIDATE.finditer(text)):
        return True
    for match in PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 15:
            return True
    return False


def score_text(text: str) -> tuple[float, dict[str, float]]:
    tokens = [w.casefold() for w in WORDS.findall(text)]
    unique = len(set(tokens)) / max(1, len(tokens))
    compact = [c for c in text if not c.isspace()]
    alnum = sum(c.isalnum() for c in compact) / max(1, len(compact))
    meaningful_lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    most_common = max(Counter(meaningful_lines).values(), default=0)
    line_diversity = 1.0 if len(meaningful_lines) <= 1 else 1 - (
        (most_common - 1) / len(meaningful_lines)
    )
    length = min(1.0, len(text) / 250.0)
    symbol_frac = 1 - alnum
    formatting = 1 - min(1.0, max(0.0, symbol_frac - 0.30) / 0.50)
    score = (0.29 * unique + 0.23 * alnum + 0.18 * line_diversity +
             0.20 * length + 0.10 * formatting)
    features = {
        "lexical_diversity": round(unique, 4),
        "alnum_fraction": round(alnum, 4),
        "line_diversity": round(line_diversity, 4),
        "length_coverage": round(length, 4),
        "formatting": round(formatting, 4),
    }
    return round(score, 4), features


def shingles(text: str, n: int = 5) -> frozenset[str]:
    words = [w.casefold() for w in WORDS.findall(text)]
    if not words:
        return frozenset()
    if len(words) < n:
        return frozenset({" ".join(words)})
    return frozenset(" ".join(words[i:i+n]) for i in range(len(words)-n+1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


class NearIndex:
    """Disk-backed MinHash/LSH candidate retrieval with exact Jaccard verification.

    False negatives are possible at the banding stage; do not describe it as
    complete near-duplicate elimination or use as a compliance guarantee.
    """

    def __init__(self, path: Path, seed: int):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            CREATE TABLE documents (sha TEXT PRIMARY KEY, shingle_json TEXT NOT NULL);
            CREATE TABLE band_index (band INTEGER, key TEXT, sha TEXT);
            CREATE INDEX band_lookup ON band_index(band, key);
            CREATE TABLE seen_ids (id TEXT PRIMARY KEY);
        """)
        rng = random.Random(seed)
        self.params = [(rng.randrange(1, MINHASH_PRIME),
                        rng.randrange(0, MINHASH_PRIME)) for _ in range(32)]

    def close(self) -> None:
        self.db.close()

    def signature(self, items: frozenset[str]) -> tuple[int, ...]:
        values = [int.from_bytes(sha256(s.encode()).digest()[:8], "big") %
                  MINHASH_PRIME for s in items]
        return tuple(min(((a * v + b) % MINHASH_PRIME for v in values),
                         default=MINHASH_PRIME) for a, b in self.params)

    @staticmethod
    def bands(signature: tuple[int, ...]):
        # 8 bands x 4 hashes, reproducible cross-runtime.
        for i in range(8):
            chunk = signature[4*i:4*i+4]
            key = digest("|".join(str(v) for v in chunk).encode())[:24]
            yield i, key

    def check(self, sha: str, items: frozenset[str],
              threshold: float) -> str | None:
        if self.db.execute("SELECT 1 FROM documents WHERE sha=?", (sha,)).fetchone():
            return "duplicate_exact"
        if not items:
            return None
        candidates: set[str] = set()
        for band, key in self.bands(self.signature(items)):
            candidates.update(row[0] for row in self.db.execute(
                "SELECT sha FROM band_index WHERE band=? AND key=?", (band, key)))
        for other in sorted(candidates):
            row = self.db.execute(
                "SELECT shingle_json FROM documents WHERE sha=?", (other,)).fetchone()
            if row and jaccard(items, frozenset(json.loads(row[0]))) >= threshold:
                return "duplicate_near"
        return None

    def add(self, sha: str, items: frozenset[str]) -> None:
        self.db.execute(
            "INSERT INTO documents(sha, shingle_json) VALUES (?,?)",
            (sha, json.dumps(sorted(items), ensure_ascii=False)))
        for band, key in self.bands(self.signature(items)):
            self.db.execute(
                "INSERT INTO band_index(band,key,sha) VALUES (?,?,?)",
                (band, key, sha))
        self.db.commit()

    def register_id(self, doc_id: str) -> bool:
        try:
            self.db.execute("INSERT INTO seen_ids(id) VALUES (?)", (doc_id,))
            self.db.commit()
            return True
        except sqlite3.IntegrityError:
            return False


def load_policy(path: Path) -> dict[str, dict]:
    """Schema v1 legacy or v2 explicit attestation (not automatic legal proof)."""
    return parse_policy(path.read_bytes())


def parse_policy(data: bytes) -> dict[str, dict]:
    try:
        config = strict_json(data.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid source policy JSON") from exc
    if (
        not isinstance(config, dict)
        or type(config.get("schema_version")) is not int
        or config["schema_version"] not in (1, 2)
        or not isinstance(config.get("sources"), list)
    ):
        raise ValueError("policy must have schema_version 1 or 2 and sources array")
    version = config["schema_version"]
    sources = {}
    for item_number, entry in enumerate(config["sources"], 1):
        if not isinstance(entry, dict):
            raise ValueError("each source policy must be an object")
        source_id = entry.get("source_id")
        if (not isinstance(source_id, str) or not source_id.strip() or
                source_id in sources or not isinstance(entry.get("license"), str) or
                entry.get("rights_basis") not in (
                    "owned", "licensed", "public_domain", "open_dataset_approved"
                ) or type(entry.get("training_allowed")) is not bool):
            raise ValueError(f"invalid or duplicate policy source at entry {item_number}")
        if version == 2 and entry["training_allowed"]:
            review = entry.get("review")
            if (not isinstance(review, dict) or
                    not isinstance(review.get("reviewed_by"), str) or
                    not review["reviewed_by"].strip() or
                    not isinstance(review.get("reviewed_at"), str) or
                    not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review["reviewed_at"]) or
                    not isinstance(review.get("evidence_sha256"), str) or
                    not re.fullmatch(r"[0-9a-f]{64}", review["evidence_sha256"]) or
                    type(entry.get("commercial_allowed")) is not bool or
                    type(entry.get("redistribution_allowed")) is not bool or
                    not isinstance(entry.get("allowed_uses"), list) or
                    "pretraining" not in entry["allowed_uses"] or
                    not all(isinstance(x, str) for x in entry["allowed_uses"]) or
                    "retention_until" not in entry):
                raise ValueError(f"v2 approved source at entry {item_number} needs review evidence, scope and retention")
            date.fromisoformat(review["reviewed_at"])
            expiry = entry["retention_until"]
            if expiry is not None:
                if not isinstance(expiry, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expiry):
                    raise ValueError(f"invalid retention_until at entry {item_number}")
                date.fromisoformat(expiry)
        sources[source_id] = {**entry, "_schema_version": version}
    return sources


def read_denylist(path: Path | None) -> tuple[set[str], set[str]]:
    """Reject malformed or empty takedown registries rather than ignoring them."""
    if path is None:
        return set(), set()
    from .lineage import read_tombstones
    return read_tombstones(path)


def read_holdout(path: Path | None, limit: int) -> tuple[set[str], list[frozenset[str]]]:
    exact: set[str] = set()
    near: list[frozenset[str]] = []
    if path is None:
        return exact, near
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= limit:
                raise ValueError("holdout record cap exceeded")
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError("holdout record requires text")
            text = canonical_text(record["text"])
            exact.add(fingerprint(text))
            near.append(shingles(text))
    return exact, near


def choose_split(group: str, seed: int) -> str:
    bucket = int(digest(f"{seed}:{group}".encode())[:12], 16) % 10_000
    if bucket < SPLIT_BUCKETS["train"]:
        return "train"
    if bucket < SPLIT_BUCKETS["train"] + SPLIT_BUCKETS["validation"]:
        return "validation"
    return "test"


def write_jsonl(stream, record: dict) -> None:
    stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")) + "\n")


def curate_dataset(input_path: Path, policy_path: Path, output_dir: Path, *,
                   config: CurationConfig | None = None,
                   holdout_path: Path | None = None,
                   denylist_path: Path | None = None,
                   language_identifier=None) -> dict:
    """Create an immutable local output directory or fail without publishing.

    Assumes an exclusive writer and an ordered input stream. Equal bytes and
    parameters produce equal output bytes; changing input order may change
    which near-duplicate representative is retained.
    """
    config = config or CurationConfig()
    input_path, policy_path, output_dir = (Path(input_path), Path(policy_path),
                                           Path(output_dir))
    if not input_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError("input JSONL and source policy must both exist")
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    lid_meta = None
    if language_identifier is not None:
        lid_meta = language_identifier.metadata()
        if (not isinstance(lid_meta, dict) or
                lid_meta.get("family") != "fasttext-lid176" or
                lid_meta.get("license_declared") != "CC-BY-SA-3.0" or
                not isinstance(lid_meta.get("model_sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", lid_meta["model_sha256"])):
            raise ValueError("language identifier requires a pinned licensed model")
    policy_bytes = policy_path.read_bytes()
    policy_sha256 = digest(policy_bytes)
    sources = parse_policy(policy_bytes)
    if config.commercial_use and any(
        p["_schema_version"] != 2 and p["training_allowed"]
        for p in sources.values()
    ):
        raise ValueError("--commercial-use requires schema v2 approved source policies")
    if any(policy.get("_schema_version") == 2 and policy["training_allowed"]
           for policy in sources.values()) and config.as_of_date is None:
        raise ValueError("schema v2 requires an explicit --as-of-date for reproducible rights expiry checks")
    holdout_sha256 = file_digest(holdout_path)
    denylist_sha256 = file_digest(denylist_path)
    denied_ids, denied_hashes = read_denylist(denylist_path)
    holdout_exact, holdout_shingles = read_holdout(
        holdout_path, config.max_holdout_records)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ada-curate-", dir=output_dir.parent))
    stats = Counter()
    accepted_by_split = Counter()
    index = None
    input_hasher = sha256()
    try:
        index = NearIndex(stage / "scratch.sqlite", config.seed)
        with (input_path.open("rb") as source,
              (stage / "train.jsonl").open("w", encoding="utf-8") as train,
              (stage / "validation.jsonl").open("w", encoding="utf-8") as validation,
              (stage / "test.jsonl").open("w", encoding="utf-8") as test,
              (stage / "audit.jsonl").open("w", encoding="utf-8") as audit):
            streams = {"train": train, "validation": validation, "test": test}
            for number, (raw_line, line_hash) in enumerate(
                iter_hashed_lines(source, config.max_chars * 20, input_hasher=input_hasher),
                1,
            ):
                stats["lines"] += 1
                source_id = None
                reason = None
                try:
                    if raw_line is None:
                        raise ValueError("record_too_large")
                    line = raw_line.decode("utf-8")
                    if len(line) > config.max_chars * 5:
                        raise ValueError("record_too_large")
                    row = strict_json(line)
                    if not isinstance(row, dict):
                        raise ValueError("invalid_record")
                    source_id = row.get("source_id")
                    rec_id, raw = row.get("id"), row.get("text")
                    policy = sources.get(source_id) if isinstance(source_id, str) else None
                    if policy is None or not policy["training_allowed"]:
                        reason = "source_not_approved"
                    elif config.commercial_use and not policy.get("commercial_allowed", False):
                        reason = "source_commercial_rights_not_granted"
                    elif policy.get("_schema_version") == 2 and (
                        policy["review"]["reviewed_at"] > config.as_of_date or
                        (policy["retention_until"] is not None and
                         policy["retention_until"] < config.as_of_date)
                    ):
                        reason = "source_rights_not_current"
                    elif (row.get("license") != policy["license"] or
                          row.get("rights_basis") != policy["rights_basis"]):
                        reason = "provenance_mismatch"
                    elif row.get("opt_out") is True or row.get("do_not_train") is True:
                        reason = "record_opt_out"
                    elif any(key in row and type(row[key]) is not bool for key in ("opt_out", "do_not_train")):
                        reason = "invalid_optout_flag"
                    elif not isinstance(rec_id, str) or not rec_id.strip() or not isinstance(raw, str):
                        reason = "invalid_record"
                    elif len(raw) > config.max_chars:
                        reason = "too_long"
                    else:
                        doc_id = f"{source_id}:{rec_id}"
                        text = canonical_text(raw)
                        sha = fingerprint(text)
                        if doc_id in denied_ids or sha in denied_hashes:
                            reason = "denylisted"
                        elif not index.register_id(doc_id):
                            reason = "duplicate_document_id"
                        elif len(text) < config.min_chars or len(WORDS.findall(text)) < config.min_words:
                            reason = "too_short"
                        elif has_pii_or_secret(text):
                            reason = "pii_or_secret"
                        elif row.get("language") not in config.supported_languages:
                            reason = "unsupported_or_undeclared_language"
                        elif config.script_sanity and not script_consistent(text, row["language"])[0]:
                            reason = "script_mismatch"
                        elif language_identifier is not None and (
                            lid_rejection := accepted_prediction(
                                language_identifier.predict(text),
                                row["language"],
                                config.lid_min_confidence,
                            )
                        ) is not None:
                            reason = lid_rejection
                        else:
                            url = clean_url(row.get("url"))
                            if (has_pii_or_secret(doc_id) or
                                    has_pii_or_secret(str(row.get("family_id", ""))) or
                                    (url is not None and has_pii_or_secret(url))):
                                raise ValueError("pii_or_secret_metadata")
                            score, features = score_text(text)
                            if score < config.min_quality:
                                reason = "low_quality"
                            else:
                                items = shingles(text)
                                if sha in holdout_exact or any(
                                    jaccard(items, h) >= config.min_near_jaccard
                                    for h in holdout_shingles
                                ):
                                    reason = "benchmark_contamination"
                                else:
                                    reason = index.check(sha, items, config.min_near_jaccard)
                                if reason is None:
                                    index.add(sha, items)
                                    family = row.get("family_id")
                                    if not isinstance(family, str) or not family.strip():
                                        family = rec_id
                                    split = choose_split(f"{source_id}:{family}", config.seed)
                                    output = {
                                        "id": doc_id,
                                        "text": text,
                                        "sha256": sha,
                                        "source_id": source_id,
                                        "license": policy["license"],
                                        "rights_basis": policy["rights_basis"],
                                        "source_url": url,
                                        "language_declared": row["language"],
                                        "family_id": family,
                                        "split": split,
                                        "quality_score": score,
                                        "domain_declared": (
                                            row.get("domain") if row.get("domain") in
                                            ("science", "code", "biomedicine", "education", "general", "other")
                                            else "unlabeled"
                                        ),
                                        "quality_features": features,
                                        "curator_version": VERSION,
                                    }
                                    write_jsonl(streams[split], output)
                                    accepted_by_split[split] += 1
                                    stats["accepted"] += 1
                except UnicodeDecodeError:
                    reason = "invalid_utf8"
                except (ValueError, TypeError, RecursionError) as exc:
                    reason = str(exc) if str(exc) in (
                        "record_too_large", "invalid_record", "invalid_url",
                        "too_long", "pii_or_secret_metadata") else "invalid_json_or_metadata"
                if reason is not None:
                    stats[f"rejected_{reason}"] += 1
                    write_jsonl(audit, {
                        "line": number,
                        "input_line_sha256": line_hash,
                        "source_id": source_id if isinstance(source_id, str) and source_id in sources else None,
                        "decision": "rejected",
                        "reason": reason,
                    })
        index.close()
        index = None
        (stage / "scratch.sqlite").unlink()
        input_sha256 = input_hasher.hexdigest()
        for path, expected, label in (
            (input_path, input_sha256, "input"),
            (policy_path, policy_sha256, "policy"),
            (holdout_path, holdout_sha256, "holdout"),
            (denylist_path, denylist_sha256, "denylist"),
        ):
            if file_digest(path) != expected:
                raise ValueError(f"{label} changed during curation")
        manifest = {
            "schema_version": 1,
            "curator_version": VERSION,
            "algorithm": "ADA-Provenance-Lattice-v0.3",
            "language_model": lid_meta,
            "input_sha256": input_sha256,
            "policy_sha256": policy_sha256,
            "holdout_sha256": holdout_sha256,
            "denylist_sha256": denylist_sha256,
            "configuration": asdict(config),
            "totals": dict(sorted(stats.items())),
            "accepted_splits": {s: accepted_by_split[s] for s in streams},
            "outputs_sha256": {name: file_digest(stage / name) for name in
                               ("train.jsonl", "validation.jsonl", "test.jsonl",
                                "audit.jsonl")},
            "limits": {
                "language": "optional pinned fastText lid.176 gate; externally calibrated accuracy required",
                "near_dedup": "MinHash-LSH candidate search; false negatives possible",
                "quality": "heuristics only; domain/model quality not certified",
                "input_order": "stable only for identical ordered inputs",
                "review": "operator must review source rights and audit findings",
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if os.path.lexists(output_dir):
            raise FileExistsError(f"refusing to overwrite {output_dir}")
        os.rename(stage, output_dir)
        return manifest
    finally:
        if index is not None:
            index.close()
        if stage.exists():
            shutil.rmtree(stage)
