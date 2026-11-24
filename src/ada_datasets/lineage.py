"""ADA Curator snapshot revocation and content-free deletion lineage (v0.2).

The operation creates a NEW validated successor snapshot, never alters a
published snapshot in place. The operator remains responsible for quarantining
or deleting ancestor snapshots, copies, training caches and downstream models.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

_SPLITS = ("train", "validation", "test")
_HEX_SHA = re.compile(r"[0-9a-f]{64}\Z")


def read_tombstones(path: Path) -> tuple[set[str], set[str]]:
    """Strict fail-closed parser: each nonblank line must name an ID or digest."""
    path = Path(path)
    identifiers: set[str] = set()
    hashes: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid tombstone JSON on line {number}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"invalid tombstone object on line {number}")
            doc_id = item.get("doc_id")
            digest = item.get("sha256")
            if doc_id is None and digest is None:
                raise ValueError(f"tombstone missing doc_id/sha256 on line {number}")
            if doc_id is not None:
                if not isinstance(doc_id, str) or not doc_id or len(doc_id) > 1024:
                    raise ValueError(f"invalid tombstone doc_id on line {number}")
                identifiers.add(doc_id)
            if digest is not None:
                if not isinstance(digest, str) or not _HEX_SHA.fullmatch(digest):
                    raise ValueError(f"invalid tombstone sha256 on line {number}")
                hashes.add(digest)
    if not identifiers and not hashes:
        raise ValueError("tombstone registry is empty")
    return identifiers, hashes


def revoke_snapshot(parent: Path, tombstones: Path, output: Path) -> dict:
    """Materialize a hash-linked successor with removed documents excluded.

    No raw document text or deletion request fields enter the appended audit.
    Excluded content remains in the *parent* directory until separately erased.
    """
    from .cli import validate_output
    from .core import VERSION, file_digest, write_jsonl

    parent, tombstones, output = Path(parent), Path(tombstones), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    validate_output(parent)  # never trust a corrupt predecessor
    denied_ids, denied_hashes = read_tombstones(tombstones)
    original = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    parent_digest = file_digest(parent / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ada-revoke-", dir=output.parent))
    removed = Counter()
    accepted = Counter()
    try:
        for split in _SPLITS:
            with ((parent / f"{split}.jsonl").open(encoding="utf-8") as source,
                  (stage / f"{split}.jsonl").open("w", encoding="utf-8") as destination,
                  (stage / "revoked.jsonl").open("a", encoding="utf-8") as receipt):
                for line in source:
                    row = json.loads(line)
                    if row["id"] in denied_ids or row["sha256"] in denied_hashes:
                        removed[split] += 1
                        # Fingerprint the serialized original line, not the ID.
                        write_jsonl(receipt, {
                            "decision": "removed",
                            "split": split,
                            "row_sha256": sha256(line.encode("utf-8")).hexdigest(),
                        })
                    else:
                        destination.write(line)
                        accepted[split] += 1
        # Keep prior rejected-row audit and append content-free removals.
        shutil.copyfile(parent / "audit.jsonl", stage / "audit.jsonl")
        with (stage / "audit.jsonl").open("a", encoding="utf-8") as audit, (
                stage / "revoked.jsonl").open(encoding="utf-8") as receipt:
            for line in receipt:
                entry = json.loads(line)
                write_jsonl(audit, {
                    "decision": "rejected",
                    "reason": "revoked_after_curation",
                    "input_line_sha256": entry["row_sha256"],
                    "source_id": None,
                })
        manifest = deepcopy(original)
        manifest["curator_version"] = VERSION
        totals = manifest.setdefault("totals", {})
        removed_total = sum(removed.values())
        totals["accepted"] = sum(accepted.values())
        totals["rejected_revoked_after_curation"] = (
            totals.get("rejected_revoked_after_curation", 0) + removed_total)
        manifest["accepted_splits"] = {name: accepted[name] for name in _SPLITS}
        manifest["lineage"] = {
            "operation": "revocation",
            "parent_manifest_sha256": parent_digest,
            "tombstone_sha256": file_digest(tombstones),
            "removed_total": removed_total,
            "removed_splits": {name: removed[name] for name in _SPLITS},
            "caution": (
                "Parent snapshots, external mirrors and previously trained "
                "models are NOT modified by this operation."
            ),
        }
        manifest["outputs_sha256"] = {
            name: file_digest(stage / name)
            for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "audit.jsonl")
        }
        manifest["revocation_receipt_sha256"] = file_digest(stage / "revoked.jsonl")
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_output(stage)
        os.rename(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)
