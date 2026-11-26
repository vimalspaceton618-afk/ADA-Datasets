"""Pinned shard intake and provenance, privacy, quality and benchmark gates."""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from hashlib import sha256
from itertools import islice
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata
from urllib.parse import unquote

from .core import (
    CurationConfig, VERSION, canonical_text, clean_url, digest, file_digest,
    fingerprint, has_pii_or_secret, iter_hashed_lines, parse_policy, read_denylist,
    score_text, strict_json, write_jsonl,
)
from .language import script_consistent
from .lid import FastTextLID176, accepted_prediction

_OPAQUE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_UNICODE_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[^\W\d_]{2,}(?!\w)")
_PAN = re.compile(r"(?<![A-Z0-9])[A-Z]{3}[ABCFGHLJPT][A-Z]\d{4}[A-Z](?![A-Z0-9])")
_SECRET = re.compile(
    r"(?<![A-Za-z0-9])(?:AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|"
    r"AIza[A-Za-z0-9_-]{35}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,})"
)
_DOMAIN_TAGS = ("science", "code", "biomedicine", "education", "general", "other")
_RELEASE_USES = ("internal", "commercial", "redistribution")
_PASSAGE_WORDS = 13


class _Reject(ValueError):
    """An expected record rejection carrying a content-free reason code."""


def _valid_id(value: object) -> bool:
    return isinstance(value, str) and _OPAQUE_ID.fullmatch(value) is not None


def release_has_pii(text: str) -> bool:
    """Identify common multilingual structured identifiers and secret patterns."""
    normalized = unicodedata.normalize("NFKC", text)
    return bool(
        has_pii_or_secret(normalized) or _UNICODE_EMAIL.search(normalized)
        or _PAN.search(normalized) or _SECRET.search(normalized)
    )


def quality_model_input(text: str) -> str:
    """Normalize fastText input and prevent its control tokens becoming labels."""
    tokens = (
        f"word-{token}" if token.startswith("__label__") or token == "</s>" else token
        for token in text.split()
    )
    return " ".join(tokens)[:50_000]


class FastTextQuality:
    """A local, checksum-pinned binary quality classifier with reviewed terms."""

    _LABELS = {"__label__keep", "__label__drop"}

    def __init__(self, model_path: Path, *, expected_sha256: str, declared_license: str):
        path = Path(model_path)
        if (
            not isinstance(expected_sha256, str) or not _SHA256.fullmatch(expected_sha256)
            or not isinstance(declared_license, str) or not declared_license.strip()
            or len(declared_license) > 200 or release_has_pii(declared_license)
            or path.suffix not in (".bin", ".ftz") or path.is_symlink() or not path.is_file()
        ):
            raise ValueError("quality model needs a local .bin/.ftz, SHA-256 and reviewed license")
        if file_digest(path) != expected_sha256:
            raise ValueError("quality model SHA-256 mismatch")
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("optional dependency missing: pip install fasttext") from exc
        self._model = fasttext.load_model(str(path))
        if file_digest(path) != expected_sha256:
            raise ValueError("quality model changed while loading")
        if set(self._model.get_labels()) != self._LABELS:
            raise ValueError("quality model must use __label__keep and __label__drop")
        self.model_sha256 = expected_sha256
        self.license_declared = declared_license

    def keep_probability(self, text: str) -> float:
        cleaned = quality_model_input(text)
        if not cleaned:
            raise ValueError("quality model input is empty")
        labels, values = self._model.predict(cleaned, k=2)
        scores = {label: float(value) for label, value in zip(labels, values)}
        # fastText can add a small epsilon at prediction time.
        if (
            len(labels) != 2 or len(values) != 2 or set(scores) != self._LABELS
            or any(not math.isfinite(value) or not 0 <= value <= 1.0001 for value in scores.values())
        ):
            raise ValueError("quality model returned an invalid prediction")
        return min(scores["__label__keep"], 1.0)

    def metadata(self) -> dict:
        return {
            "family": "fasttext-binary-quality",
            "model_sha256": self.model_sha256,
            "license_declared": self.license_declared,
        }


def _object(raw: bytes | None) -> dict:
    if raw is None:
        raise _Reject("record_too_large")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _Reject("invalid_utf8") from exc
    try:
        row = strict_json(decoded)
    except (ValueError, RecursionError) as exc:
        raise _Reject("invalid_json_or_metadata") from exc
    if not isinstance(row, dict):
        raise _Reject("invalid_record")
    return row


def _words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char if char.isalnum() or char == "_" or unicodedata.category(char).startswith("M") else " "
        for char in normalized
    ).split()


def _window_hashes(words: list[str], width: int):
    for start in range(len(words) - width + 1):
        yield sha256(" ".join(words[start:start + width]).encode("utf-8")).digest()


def _term_hashes(words: list[str]) -> set[bytes]:
    return set(_window_hashes(words, min(5, len(words)))) if words else set()


def _chunks(values):
    iterator = iter(values)
    while chunk := tuple(islice(iterator, 400)):
        yield chunk


class _HoldoutIndex:
    def __init__(self, db: sqlite3.Connection, path: Path, config: CurationConfig):
        self.db = db
        self.db.executescript("""
            CREATE TABLE benchmarks (
                row_number INTEGER PRIMARY KEY,
                content_sha256 TEXT NOT NULL UNIQUE,
                shingle_count INTEGER NOT NULL
            );
            CREATE TABLE passages (hash BLOB PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE shingles (
                hash BLOB NOT NULL,
                row_number INTEGER NOT NULL,
                PRIMARY KEY(hash, row_number)
            ) WITHOUT ROWID;
        """)
        with path.open("rb") as stream, self.db:
            for number, (raw, _) in enumerate(iter_hashed_lines(stream, config.max_chars * 20), 1):
                if number > config.max_holdout_records:
                    raise ValueError("holdout record cap exceeded")
                try:
                    row = _object(raw)
                    if not isinstance(row.get("text"), str) or len(row["text"]) > config.max_chars:
                        raise _Reject("invalid_holdout_text")
                    text = canonical_text(row["text"])
                    words = _words(text)
                    if not words:
                        raise _Reject("invalid_holdout_text")
                except _Reject as exc:
                    raise ValueError(f"invalid holdout record at line {number}") from exc
                terms = _term_hashes(words)
                inserted = self.db.execute(
                    "INSERT OR IGNORE INTO benchmarks VALUES (?, ?, ?)",
                    (number, fingerprint(text), len(terms)),
                )
                if inserted.rowcount:
                    self.db.executemany(
                        "INSERT INTO shingles VALUES (?, ?)", ((key, number) for key in terms)
                    )
                    self.db.executemany(
                        "INSERT OR IGNORE INTO passages VALUES (?)",
                        ((key,) for key in _window_hashes(words, _PASSAGE_WORDS)),
                    )

    def check(self, sha: str, words: list[str], threshold: float) -> str | None:
        if self.db.execute(
            "SELECT 1 FROM benchmarks WHERE content_sha256 = ?", (sha,)
        ).fetchone():
            return "benchmark_contamination"
        for chunk in _chunks(_window_hashes(words, _PASSAGE_WORDS)):
            placeholders = ",".join("?" for _ in chunk)
            if self.db.execute(
                f"SELECT 1 FROM passages WHERE hash IN ({placeholders}) LIMIT 1", chunk
            ).fetchone():
                return "benchmark_passage_contamination"
        terms = _term_hashes(words)
        overlaps = Counter()
        sizes = {}
        for chunk in _chunks(terms):
            placeholders = ",".join("?" for _ in chunk)
            for number, size, shared in self.db.execute(
                f"""SELECT s.row_number, b.shingle_count, count(*)
                    FROM shingles s JOIN benchmarks b USING (row_number)
                    WHERE s.hash IN ({placeholders})
                    GROUP BY s.row_number, b.shingle_count""",
                chunk,
            ):
                overlaps[number] += shared
                sizes[number] = size
        if any(
            shared / (len(terms) + sizes[number] - shared) >= threshold
            for number, shared in overlaps.items()
        ):
            return "benchmark_contamination"
        return None


def _validate_release_policies(policies: dict[str, dict], config: CurationConfig) -> None:
    if config.as_of_date is None:
        raise ValueError("release curation requires a pinned as_of_date")
    if not policies or any(policy["_schema_version"] != 2 for policy in policies.values()):
        raise ValueError("release curation requires schema v2 source policies")
    for source_id, policy in policies.items():
        if not _valid_id(source_id) or release_has_pii(source_id):
            raise ValueError("release policy source_id must be an opaque identifier")
        if not policy["training_allowed"]:
            continue
        attribution = policy.get("attribution")
        if (
            not isinstance(attribution, str) or not attribution.strip()
            or len(attribution) > 2_000 or release_has_pii(attribution)
            or any(unicodedata.category(char)[0] == "C" for char in attribution)
        ):
            raise ValueError("approved release sources require safe attribution")
        license_name = policy["license"]
        if (
            not license_name.strip() or len(license_name) > 200 or release_has_pii(license_name)
            or any(unicodedata.category(char)[0] == "C" for char in license_name)
        ):
            raise ValueError("approved release sources require a valid license declaration")


def _load_models(recipe: dict, config: CurationConfig) -> tuple:
    language, quality, calibration = None, None, {}
    if spec := recipe.get("language_model"):
        language = FastTextLID176(
            Path(spec["path"]), expected_sha256=spec["sha256"], declared_license=spec["license"]
        )
    if spec := recipe.get("quality_model"):
        calibration = spec.get("calibration")
        if (
            not isinstance(calibration, dict)
            or set(calibration) != set(config.supported_languages)
            or any(
                type(threshold) not in (int, float) or not 0 <= threshold <= 1
                for threshold in calibration.values()
            )
        ):
            raise ValueError("quality model needs finite [0,1] thresholds for configured languages")
        quality = FastTextQuality(
            Path(spec["path"]), expected_sha256=spec["sha256"], declared_license=spec["license"]
        )
    return language, quality, calibration


def _suppression(row: dict, source_id: str, max_chars: int) -> dict:
    suppressed = {}
    if _valid_id(row.get("id")):
        suppressed["doc_id_sha256"] = digest(f"{source_id}:{row['id']}".encode("utf-8"))
    raw = row.get("text")
    if isinstance(raw, str) and len(raw) <= max_chars and (text := canonical_text(raw)):
        suppressed["sha256"] = fingerprint(text)
    return suppressed


def _eligible_policy(row: dict, policy: dict) -> None:
    if not policy.get("training_allowed"):
        raise _Reject("source_not_approved")
    if row.get("license") != policy["license"] or row.get("rights_basis") != policy["rights_basis"]:
        raise _Reject("provenance_mismatch")


def _record_eligibility(
    row: dict, policy: dict, source_id: str, config: CurationConfig, release_use: str,
    denied_hashes: set[str],
) -> tuple[str, str, str, str, str | None, list[str]]:
    if (release_use == "commercial" or config.commercial_use) and not policy["commercial_allowed"]:
        raise _Reject("source_commercial_rights_not_granted")
    if release_use == "redistribution" and not policy["redistribution_allowed"]:
        raise _Reject("source_redistribution_rights_not_granted")
    if (
        policy["review"]["reviewed_at"] > config.as_of_date
        or (policy["retention_until"] is not None and policy["retention_until"] < config.as_of_date)
    ):
        raise _Reject("source_rights_not_current")
    if any(key in row and type(row[key]) is not bool for key in ("opt_out", "do_not_train")):
        raise _Reject("invalid_optout_flag")
    raw, identifier = row.get("text"), row.get("id")
    family = row.get("family_id", identifier)
    if not _valid_id(identifier) or not _valid_id(family) or not isinstance(raw, str):
        raise _Reject("invalid_record")
    if len(raw) > config.max_chars:
        raise _Reject("too_long")
    doc_id, text = f"{source_id}:{identifier}", canonical_text(raw)
    sha = fingerprint(text)
    if sha in denied_hashes:
        raise _Reject("denylisted")
    words = _words(text)
    if len(text) < config.min_chars or len(words) < config.min_words:
        raise _Reject("too_short")
    if release_has_pii(text):
        raise _Reject("pii_or_secret")
    declared = row.get("language")
    if declared not in config.supported_languages:
        raise _Reject("unsupported_or_undeclared_language")
    if config.script_sanity and not script_consistent(text, declared)[0]:
        raise _Reject("script_mismatch")
    try:
        url = clean_url(row.get("url"))
    except ValueError as exc:
        raise _Reject("invalid_url") from exc
    if (
        release_has_pii(doc_id) or release_has_pii(family)
        or (url is not None and (len(url) > 4_096 or release_has_pii(unquote(url))))
    ):
        raise _Reject("pii_or_secret_metadata")
    return doc_id, family, text, sha, url, words


def prepare_inputs(
    input_path: Path, output: Path, *, recipe: dict, policy_path: Path,
    holdout_path: Path | None, denylist_path: Path | None, input_sha256: str,
) -> dict:
    """Gate a pinned shard, producing private candidates, audit and opt-out hashes."""
    config = CurationConfig(**recipe["curation"])
    release_use = recipe["release_use"]
    if release_use not in _RELEASE_USES:
        raise ValueError("invalid release use")
    if not isinstance(input_sha256, str) or not _SHA256.fullmatch(input_sha256):
        raise ValueError("release intake requires a pinned input SHA-256")
    if os.path.lexists(output):
        raise FileExistsError("release intake output already exists")
    policy_bytes = policy_path.read_bytes()
    policy_sha = digest(policy_bytes)
    holdout_sha, denylist_sha = map(file_digest, (holdout_path, denylist_path))
    policies = parse_policy(policy_bytes)
    _validate_release_policies(policies, config)
    language_model, quality_model, calibration = _load_models(recipe, config)
    denied_ids, denied_hashes = read_denylist(denylist_path)
    counts, reasons = Counter(lines=0, candidates=0, rejected=0), Counter()
    input_hasher = sha256()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".ada-intake-", dir=output.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        with (
            closing(sqlite3.connect(Path(temporary) / "holdout.sqlite")) as db,
            input_path.open("rb") as source,
            (stage / "candidates.jsonl").open("w", encoding="utf-8") as candidates,
            (stage / "audit.jsonl").open("w", encoding="utf-8") as audit,
            (stage / "suppression.jsonl").open("w", encoding="utf-8") as suppression,
        ):
            holdout = _HoldoutIndex(db, holdout_path, config) if holdout_path is not None else None
            for number, (raw, line_hash) in enumerate(
                iter_hashed_lines(source, config.max_chars * 20, input_hasher=input_hasher), 1
            ):
                counts["lines"] += 1
                source_id = None
                try:
                    row = _object(raw)
                    candidate_source = row.get("source_id")
                    if isinstance(candidate_source, str) and candidate_source in policies:
                        source_id = candidate_source
                    policy = policies.get(source_id, {})
                    _eligible_policy(row, policy)
                    if row.get("opt_out") is True or row.get("do_not_train") is True:
                        if suppressed := _suppression(row, source_id, config.max_chars):
                            write_jsonl(suppression, suppressed)
                        raise _Reject("record_opt_out")
                    identifier = row.get("id")
                    if _valid_id(identifier) and f"{source_id}:{identifier}" in denied_ids:
                        write_jsonl(suppression, _suppression(row, source_id, config.max_chars))
                        raise _Reject("denylisted")

                    doc_id, family, text, sha, url, words = _record_eligibility(
                        row, policy, source_id, config, release_use, denied_hashes
                    )
                    declared = row["language"]
                    if language_model is not None:
                        reason = accepted_prediction(
                            language_model.predict(text), declared, config.lid_min_confidence
                        )
                        if reason:
                            raise _Reject(reason)
                    score, features = score_text(text)
                    if quality_model is not None:
                        score = quality_model.keep_probability(text)
                        minimum = calibration[declared]
                    else:
                        minimum = config.min_quality
                    if score < minimum:
                        raise _Reject("low_quality")
                    if holdout is not None and (
                        reason := holdout.check(sha, words, config.min_near_jaccard)
                    ):
                        raise _Reject(reason)
                except _Reject as exc:
                    reason = str(exc)
                    counts["rejected"] += 1
                    reasons[reason] += 1
                    write_jsonl(audit, {
                        "decision": "rejected", "input_line_sha256": line_hash,
                        "line": number, "reason": reason, "source_id": source_id,
                    })
                    continue

                domain = row.get("domain")
                write_jsonl(candidates, {
                    "id": doc_id, "text": text, "sha256": sha, "source_id": source_id,
                    "license": policy["license"], "rights_basis": policy["rights_basis"],
                    "source_url": url, "language_declared": declared, "family_id": family,
                    "quality_score": score, "quality_features": features,
                    "domain_declared": domain if domain in _DOMAIN_TAGS else "unlabeled",
                    "line": number, "input_line_sha256": line_hash,
                    "attribution": policy["attribution"], "curator_version": VERSION,
                })
                counts["candidates"] += 1

        if input_hasher.hexdigest() != input_sha256 or file_digest(input_path) != input_sha256:
            raise ValueError("release input SHA-256 mismatch")
        if any(
            file_digest(path) != expected for path, expected in
            ((policy_path, policy_sha), (holdout_path, holdout_sha), (denylist_path, denylist_sha))
        ):
            raise ValueError("release policy, holdout, or denylist changed during intake")
        if os.path.lexists(output):
            raise FileExistsError("release intake output already exists")
        os.rename(stage, output)
    return {**counts, "rejection_reasons": dict(sorted(reasons.items()))}
