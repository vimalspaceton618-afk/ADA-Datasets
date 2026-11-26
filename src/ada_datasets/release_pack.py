"""Produce and independently verify the Parquet format consumed by ADA."""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import chain, zip_longest
import json
import math
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterable

from .core import file_digest, fingerprint
from .training import (
    SPECIAL_TOKENS, SPLITS, block_schema, checked_file, checked_parquet,
    document_schema, is_sha256, parquet_modules, validate_block, validate_pack_contract,
)


_ROW_GROUP_ROWS = 256


def _load_tokenizer(payload: bytes):
    try:
        import tokenizers
    except ImportError as exc:
        raise RuntimeError("training releases require ada-datasets[release]") from exc
    data = json.loads(payload)
    if (not isinstance(data, dict) or
            any(not isinstance(data.get(field), dict) or data[field].get("type") != kind
                for field, kind in (("model", "BPE"), ("pre_tokenizer", "ByteLevel"), ("decoder", "ByteLevel"))) or
            any(data.get(field) is not None for field in ("padding", "truncation", "post_processor"))):
        raise ValueError("release tokenizer must use ADA byte-level BPE without preset padding or truncation")
    tokenizer = tokenizers.Tokenizer.from_str(payload.decode("utf-8"))
    # Recognize sentinel IDs only when the packer inserts them. Authored text
    # may happen to contain the same characters and is encoded as literal bytes.
    tokenizer.encode_special_tokens = True
    special = {key: tokenizer.token_to_id(key) for key in SPECIAL_TOKENS}
    vocab_size = tokenizer.get_vocab_size()
    if (not all(type(i) is int and 0 <= i < vocab_size for i in special.values()) or
            len(set(special.values())) != len(special)):
        raise ValueError("release tokenizer requires distinct ADA special tokens")
    return tokenizer, special, vocab_size, tokenizers.__version__


def _encode(tokenizer, text: str, special: dict) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    reserved = {special[name] for name in ("<|pad|>", "<|bos|>", "<|eos|>")}
    if any(token in reserved for token in ids):
        raise ValueError("release tokenizer emitted a reserved packing token for content")
    return ids


class _Shards:
    def __init__(self, stage: Path, category: str, split: str, schema, shard_rows: int):
        self.stage = stage
        self.prefix = f"{category}/{split}"
        self.schema = schema
        self.shard_rows = shard_rows
        self.arrow, self.parquet = parquet_modules()
        self.artifacts = {}
        self.paths = []
        self.total = 0
        self._rows = []
        self._writer = None
        self._written = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None and self._rows:
                self._flush()
        finally:
            if self._writer is not None:
                self._close_shard()

    def append(self, row: dict) -> None:
        self.total += 1
        self._rows.append(row)
        if len(self._rows) == min(_ROW_GROUP_ROWS, self.shard_rows - self._written):
            self._flush()

    def _flush(self) -> None:
        if self._writer is None:
            name = f"{self.prefix}/part-{len(self.paths):06d}.parquet"
            path = self.stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = self.parquet.ParquetWriter(
                path, self.schema, compression="zstd", use_compliant_nested_type=False,
            )
            self.paths.append(name)
        table = self.arrow.Table.from_pylist(self._rows, schema=self.schema)
        self._writer.write_table(table, row_group_size=len(self._rows))
        self._written += len(self._rows)
        self._rows.clear()
        if self._written == self.shard_rows:
            self._close_shard()

    def _close_shard(self) -> None:
        self._writer.close()
        self._writer = None
        self._written = 0
        name = self.paths[-1]
        self.artifacts[name] = file_digest(self.stage / name)


def _blocks(documents, sequence_length: int, special: dict):
    """Advance by sequence_length, retaining one context token between rows."""
    width = sequence_length + 1
    pad = special["<|pad|>"]
    bos = special["<|bos|>"]
    eos = special["<|eos|>"]
    ids: list[int] = []
    lineage: list[str] = []
    for document_id, tokens in documents:
        for token in chain((bos,), tokens, (eos,)):
            ids.append(token)
            lineage.append(document_id)
            if len(ids) == width:
                yield {
                    "input_ids": ids,
                    "loss_mask": [value != bos for value in ids[1:]],
                    "document_ids": list(dict.fromkeys(lineage)),
                }
                ids, lineage = ids[-1:], lineage[-1:]
    if len(ids) > 1:
        padded = ids + [pad] * (width - len(ids))
        yield {
            "input_ids": padded,
            "loss_mask": [value not in (bos, pad) for value in padded[1:]],
            "document_ids": list(dict.fromkeys(lineage)),
        }


def _document(row: dict, split: str) -> dict:
    document = dict(row)
    features = document.pop("quality_features", None)
    if features is not None:
        document["quality_features_json"] = json.dumps(features, ensure_ascii=False, sort_keys=True, allow_nan=False)
    _validate_document(document, split)
    return document


def _validate_document(row: dict, split: str) -> None:
    required = ("id", "text", "source_id", "license", "rights_basis", "language_declared",
                "family_id", "domain_declared", "attribution")
    score = row.get("quality_score")
    if (any(not isinstance(row.get(field), str) or not row[field] for field in required) or
            row.get("split") != split or not is_sha256(row.get("sha256")) or
            fingerprint(row["text"]) != row["sha256"] or not is_sha256(row.get("group_sha256")) or
            type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("invalid curated document")
    input_identity = (row.get("input_sha256"), row.get("line"), row.get("input_line_sha256"))
    if any(value is not None for value in input_identity):
        source, line, line_digest = input_identity
        if not is_sha256(source) or type(line) is not int or line < 1 or not is_sha256(line_digest):
            raise ValueError("invalid curated document input identity")


def _strata():
    return {dimension: defaultdict(Counter) for dimension in ("source", "language", "domain")}


def _count(strata: dict, row: dict, active_targets: int, sequence_length: int) -> None:
    for name, field in (("source", "source_id"), ("language", "language_declared"), ("domain", "domain_declared")):
        counts = strata[name][row[field]]
        counts["documents"] += 1
        counts["active_targets"] += active_targets
        counts["long_documents"] += active_targets > sequence_length


def _documents_for_packing(rows, split, tokenizer, special, budget_report, counts, strata, writer, width):
    for row in rows:
        row = _document(row, split)
        ids = _encode(tokenizer, row["text"], special)
        targets = len(ids) + 1  # Content and the document's EOS are targets.
        language, domain = row["language_declared"], row["domain_declared"]
        budgets = [budget_report[key] for key in (language, f"{language}/{domain}")
                   if split == "train" and key in budget_report]
        blocked = [budget for budget in budgets if targets > budget["remaining"]]
        if blocked:
            counts["budget_excluded_documents"] += 1
            for budget in blocked:
                budget["excluded_documents"] += 1
            continue
        for budget in budgets:
            budget["used"] += targets
            budget["remaining"] -= targets
        _count(strata, row, targets, width)
        writer.append(row)
        yield row["id"], ids


def pack_release(
    stage: Path,
    rows_by_split: dict[str, Iterable[dict]],
    *,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    sequence_length: int,
    shard_rows: int,
    token_budgets: dict[str, int],
) -> dict:
    """Write a private release stage and return metadata for its manifest."""
    if (set(rows_by_split) != set(SPLITS) or type(sequence_length) is not int or sequence_length < 2 or
            type(shard_rows) is not int or shard_rows < 1 or not isinstance(token_budgets, dict) or
            any(not isinstance(key, str) or not 1 <= len(key.split("/")) <= 2 or
                not all(key.split("/")) or type(value) is not int or value < 0
                for key, value in token_budgets.items())):
        raise ValueError("invalid release packing configuration")
    stage = Path(stage)
    payload = Path(tokenizer_path).read_bytes()
    if not is_sha256(tokenizer_sha256) or sha256(payload).hexdigest() != tokenizer_sha256:
        raise ValueError("release tokenizer digest mismatch")
    tokenizer, special, vocab_size, tokenizer_version = _load_tokenizer(payload)
    tokenizer_name = "tokenizer.json"
    (stage / tokenizer_name).write_bytes(payload)
    artifacts = {tokenizer_name: tokenizer_sha256}
    arrow, _ = parquet_modules()
    splits, composition = {}, {}
    counts = Counter()
    budgets = {key: {"limit": value, "used": 0, "remaining": value, "excluded_documents": 0}
               for key, value in sorted(token_budgets.items())}
    for split in SPLITS:
        strata = _strata()
        targets = 0
        with (_Shards(stage, "documents", split, document_schema(arrow), shard_rows) as documents,
              _Shards(stage, "blocks", split, block_schema(arrow, sequence_length), shard_rows) as blocks):
            stream = _documents_for_packing(rows_by_split[split], split, tokenizer, special,
                                            budgets, counts, strata, documents, sequence_length)
            for block in _blocks(stream, sequence_length, special):
                validate_block(block, sequence_length, special, vocab_size)
                blocks.append(block)
                targets += sum(block["loss_mask"])
        artifacts.update(documents.artifacts)
        artifacts.update(blocks.artifacts)
        splits[split] = {"documents": documents.paths, "blocks": blocks.paths,
                         "document_count": documents.total, "block_count": blocks.total,
                         "active_targets": targets}
        composition[split] = strata
    return {
        "artifacts": artifacts, "splits": splits, "sequence_length": sequence_length,
        "tokenizer": {"path": tokenizer_name, "sha256": tokenizer_sha256,
                      "special_tokens": special, "vocab_size": vocab_size},
        "composition": {"splits": composition, "token_budgets": budgets,
                        "budget_excluded_documents": counts["budget_excluded_documents"],
                        "long_document_threshold_targets": sequence_length},
        "packing_runtime": {"pyarrow": arrow.__version__, "tokenizers": tokenizer_version},
    }


def _parquet_rows(directory: Path, manifest: dict, paths: list[str], schema):
    for path in paths:
        with checked_parquet(directory, path, manifest["artifacts"][path], schema) as file:
            for batch in file.iter_batches(batch_size=_ROW_GROUP_ROWS, use_threads=False):
                yield from batch.to_pylist()


def _verify_documents(rows, split, db, tokenizer, special, count, strata, width):
    for row in rows:
        _validate_document(row, split)
        try:
            db.execute("INSERT INTO docs VALUES (?, ?)", (row["id"], row["sha256"]))
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate document identity in release") from exc
        identities = (
            ("family", json.dumps((row["source_id"], row["family_id"]))),
            ("group", row["group_sha256"]),
        )
        for kind, key in identities:
            db.execute("INSERT OR IGNORE INTO groups VALUES (?, ?, ?)", (kind, key, split))
            actual, = db.execute("SELECT split FROM groups WHERE kind=? AND key=?", (kind, key)).fetchone()
            if actual != split:
                raise ValueError("document family or duplicate group spans release splits")
        count["documents"] += 1
        ids = _encode(tokenizer, row["text"], special)
        _count(strata, row, len(ids) + 1, width)
        yield row["id"], ids


def verify_pack(directory: Path, manifest: dict) -> None:
    """Verify the tokenizer, shards, split isolation, packing and aggregate stats."""
    directory = Path(directory)
    validate_pack_contract(manifest)
    tokenizer_meta = manifest["tokenizer"]
    with checked_file(directory, tokenizer_meta["path"], tokenizer_meta["sha256"]) as stream:
        tokenizer, special, vocab_size, _ = _load_tokenizer(stream.read())
    if special != tokenizer_meta["special_tokens"] or vocab_size != tokenizer_meta["vocab_size"]:
        raise ValueError("release tokenizer IDs do not match the manifest")
    width = manifest["sequence_length"]
    arrow, _ = parquet_modules()
    actual_strata = {}
    with tempfile.TemporaryDirectory(prefix="ada-verify-pack-") as temp:
        db = sqlite3.connect(Path(temp) / "index.sqlite")
        try:
            db.executescript("""
                CREATE TABLE docs(id TEXT PRIMARY KEY, sha TEXT UNIQUE NOT NULL);
                CREATE TABLE groups(kind TEXT, key TEXT, split TEXT, PRIMARY KEY(kind,key));
            """)
            for split in SPLITS:
                metadata = manifest["splits"][split]
                count, strata = Counter(), _strata()
                documents = _parquet_rows(directory, manifest, metadata["documents"], document_schema(arrow))
                actual = _parquet_rows(directory, manifest, metadata["blocks"], block_schema(arrow, width))
                doc_stream = _verify_documents(documents, split, db, tokenizer, special, count, strata, width)
                expected = _blocks(doc_stream, width, special)
                try:
                    for seen, rebuilt in zip_longest(actual, expected):
                        if seen != rebuilt:
                            raise ValueError("packed block does not match release documents")
                        validate_block(seen, width, special, vocab_size)
                        count["blocks"] += 1
                        count["active_targets"] += sum(seen["loss_mask"])
                finally:
                    documents.close()
                    actual.close()
                if (count["documents"] != metadata["document_count"] or
                        count["blocks"] != metadata["block_count"] or
                        count["active_targets"] != metadata["active_targets"]):
                    raise ValueError("split count differs from release manifest")
                actual_strata[split] = strata
        finally:
            db.close()
    composition = manifest.get("composition")
    if (not isinstance(composition, dict) or composition.get("splits") != actual_strata or
            composition.get("long_document_threshold_targets") != width):
        raise ValueError("dataset composition differs from release documents")
