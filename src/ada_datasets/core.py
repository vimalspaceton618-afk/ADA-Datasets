"""ADA Provenance Lattice v0.1: conservative, auditable JSONL curation.

Design: approved provenance -> text integrity -> PII exclusion -> hygiene/quality
-> independent benchmark decontamination -> exact/fuzzy dedup -> stable splits
-> reproducible manifest. All content remains local, with no automatic crawling.

This is a bounded single-host implementation; neither a multilingual language-ID
model nor a production-scale distributed data curation service.
"""
from __future__ import annotations

from collections import Counter, defaultdict
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
from urllib.parse import urlsplit, urlunsplit

VERSION = "0.1.0"
WORDS = re.compile(r"\w+", re.UNICODE)
EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)")
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){10,15}(?!\d)")
PRIVATE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{24,}", re.I)
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

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (self.min_chars < 1 or self.min_words < 1 or
                self.max_chars < self.min_chars or
                not 0 <= self.min_quality <= 1 or
                not 0.5 <= self.min_near_jaccard <= 1 or
                self.max_holdout_records < 1):
            raise ValueError("invalid curation thresholds")


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


def has_pii_or_secret(text: str) -> bool:
    if EMAIL.search(text) or PRIVATE.search(text):
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
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or not isinstance(config.get("sources"), list):
        raise ValueError("policy must have schema_version=1 and sources array")
    sources = {}
    for entry in config["sources"]:
        if not isinstance(entry, dict):
            raise ValueError("each source policy must be an object")
        source_id = entry.get("source_id")
        if (not isinstance(source_id, str) or not source_id.strip() or
                source_id in sources or not isinstance(entry.get("license"), str) or
                entry.get("rights_basis") not in (
                    "owned", "licensed", "public_domain", "open_dataset_approved"
                ) or type(entry.get("training_allowed")) is not bool):
            raise ValueError(f"invalid or duplicate policy source: {source_id!r}")
        sources[source_id] = entry
    return sources


def read_denylist(path: Path | None) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    if path is None:
        return ids, hashes
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row.get("doc_id"), str):
                ids.add(row["doc_id"])
            if isinstance(row.get("sha256"), str) and re.fullmatch(
                    r"[0-9a-f]{64}", row["sha256"]):
                hashes.add(row["sha256"])
    return ids, hashes


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
            if not isinstance(record.get("text"), str):
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
                   denylist_path: Path | None = None) -> dict:
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
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    sources = load_policy(policy_path)
    denied_ids, denied_hashes = read_denylist(denylist_path)
    holdout_exact, holdout_shingles = read_holdout(
        holdout_path, config.max_holdout_records)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ada-curate-", dir=output_dir.parent))
    stats = Counter()
    accepted_by_split = Counter()
    index = None
    try:
        index = NearIndex(stage / "scratch.sqlite", config.seed)
        with (input_path.open("r", encoding="utf-8") as source,
              (stage / "train.jsonl").open("w", encoding="utf-8") as train,
              (stage / "validation.jsonl").open("w", encoding="utf-8") as validation,
              (stage / "test.jsonl").open("w", encoding="utf-8") as test,
              (stage / "audit.jsonl").open("w", encoding="utf-8") as audit):
            streams = {"train": train, "validation": validation, "test": test}
            for number, line in enumerate(source, 1):
                stats["lines"] += 1
                source_id = None
                line_hash = digest(line.encode("utf-8"))
                reason = None
                try:
                    if len(line) > config.max_chars * 5:
                        raise ValueError("record_too_large")
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("invalid_record")
                    source_id = row.get("source_id")
                    rec_id, raw = row.get("id"), row.get("text")
                    policy = sources.get(source_id) if isinstance(source_id, str) else None
                    if policy is None or not policy["training_allowed"]:
                        reason = "source_not_approved"
                    elif (row.get("license") != policy["license"] or
                          row.get("rights_basis") != policy["rights_basis"]):
                        reason = "provenance_mismatch"
                    elif row.get("opt_out") is True or row.get("do_not_train") is True:
                        reason = "record_opt_out"
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
                        else:
                            url = clean_url(row.get("url"))
                            score, features = score_text(text)
                            if score < config.min_quality:
                                reason = "low_quality"
                            elif sha in holdout_exact or any(
                                jaccard(shingles(text), h) >= config.min_near_jaccard
                                for h in holdout_shingles
                            ):
                                reason = "benchmark_contamination"
                            else:
                                items = shingles(text)
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
                                        "quality_features": features,
                                        "curator_version": VERSION,
                                    }
                                    write_jsonl(streams[split], output)
                                    accepted_by_split[split] += 1
                                    stats["accepted"] += 1
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    reason = str(exc) if str(exc) in (
                        "record_too_large", "invalid_record", "invalid_url",
                        "too_long") else "invalid_json_or_metadata"
                if reason is not None:
                    stats[f"rejected_{reason}"] += 1
                    write_jsonl(audit, {
                        "line": number,
                        "input_line_sha256": line_hash,
                        "source_id": source_id if isinstance(source_id, str) else None,
                        "decision": "rejected",
                        "reason": reason,
                    })
        index.close()
        index = None
        (stage / "scratch.sqlite").unlink()
        manifest = {
            "schema_version": 1,
            "curator_version": VERSION,
            "algorithm": "ADA-Provenance-Lattice-v0.1",
            "input_sha256": file_digest(input_path),
            "policy_sha256": file_digest(policy_path),
            "holdout_sha256": file_digest(holdout_path),
            "denylist_sha256": file_digest(denylist_path),
            "configuration": asdict(config),
            "totals": dict(sorted(stats.items())),
            "accepted_splits": {s: accepted_by_split[s] for s in streams},
            "outputs_sha256": {name: file_digest(stage / name) for name in
                               ("train.jsonl", "validation.jsonl", "test.jsonl",
                                "audit.jsonl")},
            "limits": {
                "language": "declared only, NOT verified language identification",
                "near_dedup": "MinHash-LSH candidate search; false negatives possible",
                "quality": "heuristics only; domain/model quality not certified",
                "input_order": "stable only for identical ordered inputs",
                "review": "operator must review source rights and audit findings",
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.rename(stage, output_dir)
        return manifest
    finally:
        if index is not None:
            index.close()
        if stage.exists():
            shutil.rmtree(stage)
