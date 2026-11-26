from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ada_datasets.core import file_digest
from ada_datasets.release_governance import (
    artifact_impact,
    assert_release_active,
    initialize_registry,
    quarantine_release,
    record_deletion,
    register_checkpoint,
    register_copy,
    register_release,
    verify_registry,
)


def fingerprint(label):
    return sha256(label.encode()).hexdigest()


class GovernanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "status.json"
        initialize_registry(self.path)

    def test_quarantine_reaches_checkpoints_descendants_and_copies(self):
        ancestor, child, sibling, checkpoint, copy = (
            fingerprint(label) for label in ("ancestor", "child", "sibling", "checkpoint", "copy")
        )
        with self.assertRaisesRegex(ValueError, "unregistered"):
            assert_release_active(self.path, ancestor)
        register_release(self.path, ancestor)
        register_release(self.path, child, parent_sha256=ancestor)
        register_release(self.path, sibling)
        register_checkpoint(self.path, checkpoint, child)
        register_copy(self.path, copy, artifact_kind="checkpoint", artifact_sha256=checkpoint)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            record_deletion(self.path, "copy", copy, fingerprint("receipt"))

        result = quarantine_release(self.path, ancestor, fingerprint("tombstone"))
        self.assertEqual(result["releases"], sorted([ancestor, child]))
        self.assertEqual(result["checkpoints"], [checkpoint])
        self.assertEqual(result["copies"], [copy])
        assert_release_active(self.path, sibling)
        with self.assertRaisesRegex(ValueError, "quarantined"):
            assert_release_active(self.path, child)

        late_checkpoint = fingerprint("late-checkpoint")
        self.assertFalse(register_checkpoint(self.path, late_checkpoint, child)["active"])
        result = quarantine_release(self.path, ancestor, fingerprint("tombstone"))
        self.assertEqual(result["checkpoints"], sorted([checkpoint, late_checkpoint]))

        receipt = record_deletion(self.path, "copy", copy, fingerprint("receipt"))
        self.assertEqual(record_deletion(self.path, "copy", copy, fingerprint("receipt")), receipt)
        report = verify_registry(self.path, expected_head_sha256=receipt["head_sha256"])
        self.assertEqual(report["pending_deletions"]["copy"], [])
        self.assertEqual(report["deletion_receipts"][0]["receipt_sha256"], fingerprint("receipt"))
        with self.assertRaisesRegex(ValueError, "unique"):
            record_deletion(self.path, "copy", copy, fingerprint("other receipt"))

    def test_replacement_and_new_tombstone(self):
        ancestor, successor, first, second = (
            fingerprint(label) for label in ("parent", "successor", "first tombstone", "second tombstone")
        )
        register_release(self.path, ancestor)
        with self.assertRaisesRegex(ValueError, "replacement"):
            register_release(self.path, successor, parent_sha256=ancestor, replacement=True)
        quarantine_release(self.path, ancestor, first)
        publication = register_release(self.path, successor, parent_sha256=ancestor, replacement=True)
        self.assertEqual(register_release(self.path, successor, parent_sha256=ancestor, replacement=True), publication)
        self.assertEqual(quarantine_release(self.path, ancestor, first)["releases"], [ancestor])
        assert_release_active(self.path, successor)
        self.assertEqual(quarantine_release(self.path, ancestor, second)["releases"], sorted([ancestor, successor]))
        with self.assertRaisesRegex(ValueError, "quarantined"):
            assert_release_active(self.path, successor)

    def test_corruption_and_external_head_fail_closed(self):
        release = fingerprint("public release")
        status = register_release(self.path, release)
        with self.assertRaisesRegex(ValueError, "independently pinned"):
            verify_registry(self.path, expected_head_sha256=fingerprint("other head"))
        self.assertEqual(verify_registry(self.path, expected_head_sha256=status["head_sha256"])["event_count"], 1)
        data = json.loads(self.path.read_text())
        data["events"][0]["data"]["release_sha256"] = fingerprint("tampered")
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            assert_release_active(self.path, release)

    def test_concurrent_checkpoint_registration(self):
        release = fingerprint("release")
        register_release(self.path, release)
        checkpoints = [fingerprint(str(number)) for number in range(20)]
        with ThreadPoolExecutor(max_workers=5) as workers:
            list(workers.map(lambda checkpoint: register_checkpoint(self.path, checkpoint, release), checkpoints))
        self.assertEqual(quarantine_release(self.path, release, fingerprint("request"))["checkpoints"], sorted(checkpoints))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "Parquet extra required")
    def test_tombstone_matches_rejected_exact_duplicate_alias(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        directory = self.path.parent
        documents = "documents/train/part-000000.parquet"
        blocks = "blocks/train/part-000000.parquet"
        content = fingerprint("curated content")
        tables = {
            documents: pa.table({"id": ["example:representative"], "sha256": [content]}),
            blocks: pa.table({"document_ids": [["example:representative"]]}),
        }
        for name, table in tables.items():
            path = directory / name
            path.parent.mkdir(parents=True)
            pq.write_table(table, path)
        (directory / "aliases.jsonl").write_text(
            json.dumps({"doc_id_sha256": fingerprint("example:alias"), "sha256": content}) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 2,
            "format": "ada-training-release",
            "artifacts": {name: file_digest(directory / name) for name in (*tables, "aliases.jsonl")},
            "splits": {
                "train": {"documents": [documents], "blocks": [blocks]},
                "validation": {"documents": [], "blocks": []},
                "test": {"documents": [], "blocks": []},
            },
        }
        tombstones = directory / "tombstones.jsonl"
        tombstones.write_text('{"doc_id":"example:alias"}\n', encoding="utf-8")
        self.assertEqual(artifact_impact(directory, manifest, tombstones), {
            "matched_documents": 1,
            "document_artifacts": [documents],
            "block_artifacts": [blocks],
            "document_sha256": [content],
        })
