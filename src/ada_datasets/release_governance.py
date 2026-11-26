"""Private release lifecycle and deletion receipts for ADA training data."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from .core import file_digest
from .lineage import read_tombstones


_GENESIS = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_COLLECTIONS = {"release": "releases", "checkpoint": "checkpoints", "copy": "copies"}
_SPLITS = ("train", "validation", "test")


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _object(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError(f"invalid {label} fields")
    return value


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _event_sha(event: dict) -> str:
    return sha256(_json_bytes(event)).hexdigest()


@dataclass
class _Registry:
    head: str = _GENESIS
    events: list[dict] = field(default_factory=list)
    releases: dict[str, tuple[str | None, bool]] = field(default_factory=dict)
    children: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    checkpoints: dict[str, str] = field(default_factory=dict)
    copies: dict[str, tuple[str, str]] = field(default_factory=dict)
    quarantines: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    blocked: set[str] = field(default_factory=set)
    receipts: dict[tuple[str, str], str] = field(default_factory=dict)

    def descendants(self, release: str) -> set[str]:
        seen = {release}
        pending = deque([release])
        while pending:
            for child in self.children.get(pending.popleft(), ()):
                if child not in seen:
                    seen.add(child)
                    pending.append(child)
        return seen

    def owning_release(self, kind: str, artifact: str) -> str:
        if kind == "release" and artifact in self.releases:
            return artifact
        if kind == "checkpoint" and artifact in self.checkpoints:
            return self.checkpoints[artifact]
        if kind == "copy" and artifact in self.copies:
            return self.owning_release(*self.copies[artifact])
        raise ValueError("artifact is absent from the release registry")

    def impact(self, releases: set[str]) -> dict:
        return {
            "releases": sorted(releases),
            "checkpoints": sorted(key for key, release in self.checkpoints.items() if release in releases),
            "copies": sorted(key for key in self.copies if self.owning_release("copy", key) in releases),
        }

    def apply(self, kind: str, data: dict) -> None:
        if kind == "release":
            _object(data, {"release_sha256", "parent_sha256", "replacement"}, "release registration")
            release = _digest(data["release_sha256"], "release")
            parent = data["parent_sha256"]
            replacement = data["replacement"]
            if type(replacement) is not bool or release in self.releases:
                raise ValueError("invalid or repeated release registration")
            if parent is not None:
                _digest(parent, "parent release")
                if parent not in self.releases or replacement != (parent in self.blocked):
                    raise ValueError("release parent is absent or inconsistent with replacement status")
                self.children[parent].add(release)
            elif replacement:
                raise ValueError("a replacement release requires a quarantined parent")
            self.releases[release] = (parent, replacement)
        elif kind == "checkpoint":
            _object(data, {"checkpoint_sha256", "release_sha256"}, "checkpoint registration")
            checkpoint = _digest(data["checkpoint_sha256"], "checkpoint")
            release = _digest(data["release_sha256"], "release")
            if checkpoint in self.checkpoints or release not in self.releases:
                raise ValueError("invalid or repeated checkpoint registration")
            self.checkpoints[checkpoint] = release
        elif kind == "copy":
            _object(data, {"copy_sha256", "artifact_kind", "artifact_sha256"}, "copy registration")
            copy = _digest(data["copy_sha256"], "copy")
            artifact = _digest(data["artifact_sha256"], "artifact")
            artifact_kind = data["artifact_kind"]
            if copy in self.copies or artifact_kind not in ("release", "checkpoint"):
                raise ValueError("invalid or repeated copy registration")
            self.owning_release(artifact_kind, artifact)
            self.copies[copy] = (artifact_kind, artifact)
        elif kind == "quarantine":
            _object(data, {"release_sha256", "tombstone_sha256"}, "quarantine record")
            release = _digest(data["release_sha256"], "release")
            tombstones = _digest(data["tombstone_sha256"], "tombstones")
            key = (release, tombstones)
            if release not in self.releases or key in self.quarantines:
                raise ValueError("invalid or repeated quarantine record")
            affected = self.descendants(release)
            self.quarantines[key] = affected
            self.blocked.update(affected)
        elif kind == "deletion":
            _object(data, {"artifact_kind", "artifact_sha256", "receipt_sha256"}, "deletion record")
            artifact = _digest(data["artifact_sha256"], "artifact")
            receipt = _digest(data["receipt_sha256"], "deletion receipt")
            artifact_kind = data["artifact_kind"]
            if not isinstance(artifact_kind, str) or artifact_kind not in _ARTIFACT_COLLECTIONS:
                raise ValueError("invalid deletion artifact kind")
            key = (artifact_kind, artifact)
            if key in self.receipts or self.owning_release(*key) not in self.blocked:
                raise ValueError("deletion receipt requires a quarantined artifact and must be unique")
            self.receipts[key] = receipt
        else:
            raise ValueError("unknown registry event kind")

    def append(self, kind: str, data: dict) -> None:
        self.apply(kind, data)
        entry = {"kind": kind, "data": data, "previous_sha256": self.head}
        self.head = _event_sha(entry)
        self.events.append({**entry, "sha256": self.head})

    def payload(self) -> dict:
        return {"schema_version": 1, "head_sha256": self.head, "events": self.events}


def _read_registry(path: Path) -> _Registry:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("release registry must be a regular file")
        data = _object(json.load(stream), {"schema_version", "head_sha256", "events"}, "registry")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or not isinstance(data["events"], list):
        raise ValueError("unsupported release registry schema")
    registry = _Registry()
    for recorded in data["events"]:
        _object(recorded, {"kind", "data", "previous_sha256", "sha256"}, "registry event")
        if recorded["previous_sha256"] != registry.head:
            raise ValueError("release registry history is broken")
        registry.append(recorded["kind"], recorded["data"])
        if recorded["sha256"] != registry.head:
            raise ValueError("release registry event digest mismatch")
    if data["head_sha256"] != registry.head:
        raise ValueError("release registry head mismatch")
    return registry


@contextmanager
def _write_lock(path: Path):
    import fcntl

    lock = path.with_name(path.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("release registry lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _write_registry(path: Path, registry: _Registry) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".ada-status-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(registry.payload()) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_registry(status_path: Path) -> dict:
    """Create a private registry, or verify the existing registry on retry."""
    path = Path(status_path)
    with _write_lock(path):
        try:
            registry = _read_registry(path)
        except FileNotFoundError:
            registry = _Registry()
            _write_registry(path, registry)
    return {"head_sha256": registry.head, "event_count": len(registry.events)}


def verify_registry(status_path: Path, *, expected_head_sha256: str | None = None) -> dict:
    registry = _read_registry(Path(status_path))
    if expected_head_sha256 is not None and registry.head != _digest(expected_head_sha256, "expected registry head"):
        raise ValueError("release registry differs from the independently pinned head")
    quarantined = registry.impact(registry.blocked)
    return {
        "head_sha256": registry.head,
        "event_count": len(registry.events),
        "active_releases": sorted(registry.releases.keys() - registry.blocked),
        "quarantined": quarantined,
        "pending_deletions": {
            kind: [artifact for artifact in quarantined[collection] if (kind, artifact) not in registry.receipts]
            for kind, collection in _ARTIFACT_COLLECTIONS.items()
        },
        "deletion_receipts": [
            {"artifact_kind": kind, "artifact_sha256": artifact, "receipt_sha256": receipt}
            for (kind, artifact), receipt in sorted(registry.receipts.items())
        ],
    }


def assert_release_active(status_path: Path, manifest_sha256: str) -> None:
    """Fail closed when release status is unavailable, unregistered, or blocked."""
    release = _digest(manifest_sha256, "release manifest")
    registry = _read_registry(Path(status_path))
    if release not in registry.releases or release in registry.blocked:
        raise ValueError("release is unregistered or quarantined")


def register_release(
    status_path: Path,
    manifest_sha256: str,
    *,
    parent_sha256: str | None = None,
    replacement: bool = False,
) -> dict:
    """Register a verified release. Only the successor builder marks replacements."""
    release = _digest(manifest_sha256, "release manifest")
    if parent_sha256 is not None:
        _digest(parent_sha256, "parent manifest")
    if type(replacement) is not bool:
        raise ValueError("replacement must be a Boolean")
    path = Path(status_path)
    with _write_lock(path):
        registry = _read_registry(path)
        if registry.releases.get(release) != (parent_sha256, replacement):
            registry.append("release", {
                "release_sha256": release, "parent_sha256": parent_sha256, "replacement": replacement,
            })
            _write_registry(path, registry)
    return {"release_sha256": release, "head_sha256": registry.head, "active": release not in registry.blocked}


def register_checkpoint(status_path: Path, checkpoint_sha256: str, release_sha256: str) -> dict:
    """Record a checkpoint even if quarantine raced its final publication."""
    path = Path(status_path)
    checkpoint = _digest(checkpoint_sha256, "checkpoint")
    release = _digest(release_sha256, "release")
    with _write_lock(path):
        registry = _read_registry(path)
        if registry.checkpoints.get(checkpoint) != release:
            registry.append("checkpoint", {"checkpoint_sha256": checkpoint, "release_sha256": release})
            _write_registry(path, registry)
    return {"checkpoint_sha256": checkpoint, "head_sha256": registry.head, "active": release not in registry.blocked}


def register_copy(status_path: Path, copy_sha256: str, *, artifact_kind: str, artifact_sha256: str) -> dict:
    path = Path(status_path)
    copy = _digest(copy_sha256, "copy")
    artifact = _digest(artifact_sha256, "artifact")
    with _write_lock(path):
        registry = _read_registry(path)
        if registry.copies.get(copy) != (artifact_kind, artifact):
            registry.append("copy", {"copy_sha256": copy, "artifact_kind": artifact_kind, "artifact_sha256": artifact})
            _write_registry(path, registry)
    return {"copy_sha256": copy, "head_sha256": registry.head}


def quarantine_release(status_path: Path, manifest_sha256: str, tombstone_sha256: str) -> dict:
    path = Path(status_path)
    release = _digest(manifest_sha256, "release")
    tombstones = _digest(tombstone_sha256, "tombstones")
    key = (release, tombstones)
    with _write_lock(path):
        registry = _read_registry(path)
        if key not in registry.quarantines:
            registry.append("quarantine", {"release_sha256": release, "tombstone_sha256": tombstones})
            _write_registry(path, registry)
    return {"head_sha256": registry.head, **registry.impact(registry.quarantines[key])}


def record_deletion(status_path: Path, artifact_kind: str, artifact_sha256: str, receipt_sha256: str) -> dict:
    path = Path(status_path)
    artifact = _digest(artifact_sha256, "artifact")
    receipt = _digest(receipt_sha256, "deletion receipt")
    if not isinstance(artifact_kind, str) or artifact_kind not in _ARTIFACT_COLLECTIONS:
        raise ValueError("invalid deletion artifact kind")
    with _write_lock(path):
        registry = _read_registry(path)
        if registry.receipts.get((artifact_kind, artifact)) != receipt:
            registry.append("deletion", {
                "artifact_kind": artifact_kind, "artifact_sha256": artifact, "receipt_sha256": receipt,
            })
            _write_registry(path, registry)
    return {"artifact_sha256": artifact, "receipt_sha256": receipt, "head_sha256": registry.head}


def _artifact_file(directory: Path, manifest: dict, name: str, kind: str, split: str) -> Path:
    if not isinstance(name, str) or re.fullmatch(rf"{kind}/{split}/part-[0-9]{{6}}\.parquet", name) is None:
        raise ValueError("invalid release artifact path")
    path = directory / name
    pinned = _digest(manifest["artifacts"].get(name), "release artifact")
    if path.is_symlink() or path.parent.is_symlink() or path.parent.parent.is_symlink() or file_digest(path) != pinned:
        raise ValueError("release artifact digest mismatch or symlink")
    return path


def artifact_impact(directory: Path, manifest: dict, tombstone_path: Path) -> dict:
    """Locate documents and blocks directly affected by a private tombstone set."""
    try:
        from pyarrow.parquet import ParquetFile
    except ImportError as exc:
        raise ValueError("release impact requires the Parquet extra") from exc

    if manifest.get("format") != "ada-training-release" or manifest.get("schema_version") != 2:
        raise ValueError("release impact requires a v2 training release")
    splits = _object(manifest.get("splits"), set(_SPLITS), "release splits")
    if not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("invalid release artifact inventory")
    denied_ids, denied_hashes = read_tombstones(Path(tombstone_path))
    directory = Path(directory)
    matched_ids: set[str] = set()
    matched_hashes: set[str] = set()
    document_artifacts: set[str] = set()
    block_artifacts: set[str] = set()
    alias_path = directory / "aliases.jsonl"
    if "aliases.jsonl" in manifest["artifacts"]:
        pinned_aliases = _digest(manifest["artifacts"]["aliases.jsonl"], "release alias inventory")
        if alias_path.is_symlink() or not alias_path.is_file() or file_digest(alias_path) != pinned_aliases:
            raise ValueError("release alias inventory digest mismatch or symlink")
        id_hashes = {sha256(doc_id.encode("utf-8")).hexdigest() for doc_id in denied_ids}
        with alias_path.open(encoding="utf-8") as aliases:
            for item in aliases:
                alias = _object(json.loads(item), {"doc_id_sha256", "sha256"}, "release alias")
                id_hash = _digest(alias["doc_id_sha256"], "release alias identifier")
                content_hash = _digest(alias["sha256"], "release alias content")
                if id_hash in id_hashes:
                    matched_hashes.add(content_hash)
        denied_hashes.update(matched_hashes)
    for split, metadata in splits.items():
        if not isinstance(metadata, dict) or not all(isinstance(metadata.get(key), list) for key in ("documents", "blocks")):
            raise ValueError("invalid release split inventory")
        for name in metadata["documents"]:
            path = _artifact_file(directory, manifest, name, "documents", split)
            with ParquetFile(path) as parquet:
                for batch in parquet.iter_batches(columns=["id", "sha256"]):
                    for row in batch.to_pylist():
                        if row["id"] in denied_ids or row["sha256"] in denied_hashes:
                            matched_ids.add(row["id"])
                            matched_hashes.add(row["sha256"])
                            document_artifacts.add(name)

    if matched_ids:
        for split, metadata in splits.items():
            for name in metadata["blocks"]:
                path = _artifact_file(directory, manifest, name, "blocks", split)
                with ParquetFile(path) as parquet:
                    blocks = parquet.iter_batches(columns=["document_ids"])
                    if any(matched_ids.intersection(row["document_ids"]) for batch in blocks for row in batch.to_pylist()):
                        block_artifacts.add(name)

    return {
        "matched_documents": len(matched_ids),
        "document_artifacts": sorted(document_artifacts),
        "block_artifacts": sorted(block_artifacts),
        "document_sha256": sorted(matched_hashes),
    }


def add_governance_commands(subparsers) -> None:
    registry = subparsers.add_parser("governance", help="manage a private release status registry")
    registry.add_argument("--status", type=Path, required=True)
    commands = registry.add_subparsers(dest="governance_command", required=True)
    commands.add_parser("init", help="initialize the private release status registry")
    verify = commands.add_parser("verify", help="verify history and report pending quarantine work")
    verify.add_argument("--expected-head-sha256")
    register = commands.add_parser("register-release", help="register a previously verified release")
    register.add_argument("--release-sha256", required=True)
    register.add_argument("--parent-sha256")
    checkpoint = commands.add_parser("register-checkpoint", help="link a checkpoint to its input release")
    checkpoint.add_argument("--checkpoint-sha256", required=True)
    checkpoint.add_argument("--release-sha256", required=True)
    copy = commands.add_parser("register-copy", help="link an operator's private copy identifier to an artifact")
    copy.add_argument("--copy-sha256", required=True)
    copy.add_argument("--artifact-kind", choices=["release", "checkpoint"], required=True)
    copy.add_argument("--artifact-sha256", required=True)
    quarantine = commands.add_parser("quarantine", help="quarantine a release and all registered descendants")
    quarantine.add_argument("--release-sha256", required=True)
    quarantine.add_argument("--tombstone-sha256", required=True)
    receipt = commands.add_parser("receipt", help="pin a private operator deletion receipt")
    receipt.add_argument("--artifact-kind", choices=sorted(_ARTIFACT_COLLECTIONS), required=True)
    receipt.add_argument("--artifact-sha256", required=True)
    receipt.add_argument("--receipt-sha256", required=True)


def run_governance_command(args) -> dict:
    command = args.governance_command
    if command == "init":
        return initialize_registry(args.status)
    if command == "verify":
        return verify_registry(args.status, expected_head_sha256=args.expected_head_sha256)
    if command == "register-release":
        return register_release(args.status, args.release_sha256, parent_sha256=args.parent_sha256)
    if command == "register-checkpoint":
        return register_checkpoint(args.status, args.checkpoint_sha256, args.release_sha256)
    if command == "register-copy":
        return register_copy(args.status, args.copy_sha256, artifact_kind=args.artifact_kind, artifact_sha256=args.artifact_sha256)
    if command == "quarantine":
        return quarantine_release(args.status, args.release_sha256, args.tombstone_sha256)
    if command == "receipt":
        return record_deletion(args.status, args.artifact_kind, args.artifact_sha256, args.receipt_sha256)
    raise ValueError("unknown governance command")
