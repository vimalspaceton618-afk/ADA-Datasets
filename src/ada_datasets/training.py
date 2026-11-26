"""Verified, seekable access to ADA training releases on private storage."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re


SPLITS = ("train", "validation", "test")
SPECIAL_TOKENS = ("<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>")
CURSOR_KEYS = ("shard_index", "row_group", "row", "epoch")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def parquet_modules():
    try:
        import pyarrow as arrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Parquet releases require ada-datasets[release]") from exc
    return arrow, parquet


def document_schema(arrow):
    required_strings = (
        "id", "sha256", "source_id", "license", "rights_basis", "language_declared",
        "family_id", "domain_declared", "group_sha256", "split", "attribution",
    )
    return arrow.schema([
        *(arrow.field(name, arrow.string(), nullable=False) for name in required_strings),
        arrow.field("text", arrow.large_string(), nullable=False),
        arrow.field("quality_score", arrow.float64(), nullable=False),
        arrow.field("quality_features_json", arrow.string()),
        arrow.field("source_url", arrow.string()),
        arrow.field("input_sha256", arrow.string()),
        arrow.field("line", arrow.int64()),
        arrow.field("input_line_sha256", arrow.string()),
    ])


def block_schema(arrow, sequence_length: int):
    return arrow.schema([
        arrow.field("input_ids", arrow.list_(arrow.int32(), sequence_length + 1), nullable=False),
        arrow.field("loss_mask", arrow.list_(arrow.bool_(), sequence_length), nullable=False),
        arrow.field("document_ids", arrow.list_(arrow.string()), nullable=False),
    ])


def artifact_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ValueError("invalid release artifact path")
    relative = PurePosixPath(name)
    if (relative.as_posix() != name or relative.is_absolute() or
            any(part in (".", "..") for part in relative.parts)):
        raise ValueError("invalid release artifact path")
    path = directory
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("release artifacts cannot contain symlinks")
    if not path.is_file():
        raise ValueError(f"missing release artifact: {name}")
    return path


@contextmanager
def checked_file(directory: Path, name: str, expected_sha256: str):
    if not is_sha256(expected_sha256):
        raise ValueError("release artifact needs a SHA-256 digest")
    with artifact_path(directory, name).open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"release artifact digest mismatch: {name}")
        stream.seek(0)
        yield stream


@contextmanager
def checked_parquet(directory: Path, name: str, expected_sha256: str, schema):
    _, parquet = parquet_modules()
    with checked_file(directory, name, expected_sha256) as stream:
        file = parquet.ParquetFile(stream)
        try:
            if not file.schema_arrow.equals(schema):
                raise ValueError(f"release Parquet schema mismatch: {name}")
            if file.num_row_groups == 0 or not file.metadata.num_rows:
                raise ValueError("release Parquet shard is empty")
            yield file
        finally:
            file.close()


def validate_pack_contract(manifest: dict) -> None:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or
            manifest["schema_version"] != 2 or manifest.get("format") != "ada-training-release"):
        raise ValueError("unsupported training release manifest")
    width = manifest.get("sequence_length")
    tokenizer = manifest.get("tokenizer")
    artifacts = manifest.get("artifacts")
    splits = manifest.get("splits")
    if (type(width) is not int or width < 2 or not isinstance(tokenizer, dict) or
            not isinstance(artifacts, dict) or not isinstance(splits, dict) or
            set(splits) != set(SPLITS)):
        raise ValueError("invalid training release metadata")
    path = tokenizer.get("path")
    special = tokenizer.get("special_tokens")
    vocab_size = tokenizer.get("vocab_size")
    if (not isinstance(path, str) or not is_sha256(tokenizer.get("sha256")) or
            artifacts.get(path) != tokenizer["sha256"] or not isinstance(special, dict) or
            set(special) != set(SPECIAL_TOKENS) or type(vocab_size) is not int or
            not all(type(i) is int and 0 <= i < vocab_size for i in special.values()) or
            len(set(special.values())) != len(SPECIAL_TOKENS)):
        raise ValueError("invalid release tokenizer metadata")
    claimed = {path}
    for split, metadata in splits.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid {split} split metadata")
        for category, count_name in (("documents", "document_count"), ("blocks", "block_count")):
            paths, count = metadata.get(category), metadata.get(count_name)
            if (not isinstance(paths, list) or type(count) is not int or
                    count < len(paths) or bool(count) != bool(paths)):
                raise ValueError(f"invalid {split} {category} metadata")
            for name in paths:
                if (not isinstance(name, str) or name in claimed or
                        not is_sha256(artifacts.get(name))):
                    raise ValueError(f"invalid {split} {category} artifact")
                claimed.add(name)
        targets = metadata.get("active_targets")
        if (type(targets) is not int or
                not metadata["block_count"] <= targets <= metadata["block_count"] * width or
                targets < metadata["document_count"]):
            raise ValueError(f"invalid {split} target count")


def validate_block(row: dict, sequence_length: int, special: dict, vocab_size: int) -> None:
    ids, mask, docs = row["input_ids"], row["loss_mask"], row["document_ids"]
    if (not isinstance(ids, list) or len(ids) != sequence_length + 1 or
            not all(type(i) is int and 0 <= i < vocab_size for i in ids) or
            not isinstance(mask, list) or len(mask) != sequence_length or
            not all(type(bit) is bool for bit in mask) or not any(mask) or
            not isinstance(docs, list) or not docs or
            not all(isinstance(doc, str) and doc for doc in docs) or len(set(docs)) != len(docs)):
        raise ValueError("invalid packed training block")
    pad, bos = special["<|pad|>"], special["<|bos|>"]
    if pad in ids and (ids[0] == pad or any(i != pad for i in ids[ids.index(pad):])):
        raise ValueError("block padding must follow content")
    if mask != [i not in (pad, bos) for i in ids[1:]]:
        raise ValueError("block mask does not match token roles")


class TrainingRelease:
    """Read split-local batches using checkpointable Parquet row-group cursors."""

    def __init__(self, directory: Path, expected_manifest_sha256: str, *, status_path: Path | None = None):
        self.directory = Path(directory)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError("training release must be a regular directory")
        self.manifest_sha256 = expected_manifest_sha256
        self.status_path = Path(status_path) if status_path is not None else None
        with checked_file(self.directory, "manifest.json", expected_manifest_sha256) as stream:
            self.manifest = json.load(stream)
        validate_pack_contract(self.manifest)
        self.check_active()
        self.sequence_length = self.manifest["sequence_length"]
        tokenizer = self.manifest["tokenizer"]
        self.tokenizer_path = artifact_path(self.directory, tokenizer["path"])
        self.tokenizer_sha256 = tokenizer["sha256"]
        self.special_tokens = dict(tokenizer["special_tokens"])
        self.vocab_size = tokenizer["vocab_size"]
        with checked_file(self.directory, tokenizer["path"], self.tokenizer_sha256):
            pass
        self._resources = ExitStack()
        self._file = self._open_name = self._row_group = self._row_group_index = None

    def check_active(self) -> None:
        if self.status_path is not None:
            from .release_governance import assert_release_active
            assert_release_active(self.status_path, self.manifest_sha256)

    def close(self) -> None:
        self._resources.close()
        self._file = self._open_name = self._row_group = self._row_group_index = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _paths(self, split: str) -> list[str]:
        if split not in SPLITS:
            raise ValueError("unknown release split")
        return self.manifest["splits"][split]["blocks"]

    def cursor(self, split: str) -> dict[str, int]:
        self._paths(split)
        return dict.fromkeys(CURSOR_KEYS, 0)

    def _read_group(self, name: str, row_group: int):
        if name != self._open_name:
            self.close()
            arrow, _ = parquet_modules()
            self._file = self._resources.enter_context(checked_parquet(
                self.directory, name, self.manifest["artifacts"][name],
                block_schema(arrow, self.sequence_length),
            ))
            self._open_name = name
        if row_group >= self._file.num_row_groups:
            raise ValueError("cursor row group is outside the shard")
        if self._row_group_index != row_group:
            self._row_group = self._file.read_row_group(row_group, use_threads=False)
            self._row_group_index = row_group
        return self._row_group, self._file.num_row_groups

    def next_batch(self, split: str, cursor: dict, batch_size: int, *, repeat: bool = False):
        self.check_active()
        paths = self._paths(split)
        if (type(batch_size) is not int or batch_size < 1 or not isinstance(cursor, dict) or
                set(cursor) != set(CURSOR_KEYS) or any(type(v) is not int or v < 0 for v in cursor.values())):
            raise ValueError("invalid batch size or data cursor")
        shard, group, row, epoch = (cursor[key] for key in CURSOR_KEYS)
        if shard > len(paths) or (shard == len(paths) and (group or row)):
            raise ValueError("cursor position is outside the training split")
        rows = []
        while paths and len(rows) < batch_size:
            if shard == len(paths):
                if not repeat:
                    break
                shard, epoch = 0, epoch + 1
            table, group_count = self._read_group(paths[shard], group)
            if row >= table.num_rows:
                raise ValueError("cursor row is outside the row group")
            take = min(batch_size - len(rows), table.num_rows - row)
            rows.extend(table.slice(row, take).to_pylist())
            row += take
            if row == table.num_rows:
                group, row = group + 1, 0
                if group == group_count:
                    shard, group = shard + 1, 0
        if not rows:
            raise StopIteration
        for block in rows:
            validate_block(block, self.sequence_length, self.special_tokens, self.vocab_size)
        return rows, dict(zip(CURSOR_KEYS, (shard, group, row, epoch)))
