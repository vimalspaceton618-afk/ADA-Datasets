"""ADA Curator v0.2 governance, lineage, language and empirical-quality gates."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ada_datasets.cli import main, validate_output
from ada_datasets.core import CurationConfig, curate_dataset, fingerprint, load_policy
from ada_datasets.language import script_consistent, script_fractions
from ada_datasets.lineage import read_tombstones, revoke_snapshot
from ada_datasets.quality import evaluate_quality, write_report

A = (
    "An independent science course explains an experimental protocol with "
    "clear measurements, recorded observations, carefully controlled variables, "
    "and uncertainty analysis. Learners compare multiple explanations and "
    "evaluate reproducibility. Good scientific practice explicitly documents "
    "assumptions, instrument calibration, and findings in a format that other "
    "researchers can validate and review without special access."
)
B = (
    "Computer science courses illustrate how tests establish stable contracts "
    "for programs in reliable production systems. Engineers describe expected "
    "behaviors, verify exceptional circumstances, monitor error rates, and "
    "maintain structured release records. They run repeatable evaluations, "
    "separate hidden validation examples, and communicate system limitations "
    "so subsequent maintainers can investigate failures independently."
)


class V02Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = self.root / "sources.json"
        self.policy.write_text(json.dumps({
            "schema_version": 1,
            "sources": [{
                "source_id": "local",
                "license": "CC0-1.0",
                "rights_basis": "owned",
                "training_allowed": True,
            }]
        }), encoding="utf-8")
        self.input = self.root / "input.jsonl"

    def rows(self, *records):
        self.input.write_text("".join(json.dumps(row, ensure_ascii=False)
                                     + "\n" for row in records),
                              encoding="utf-8")

    @staticmethod
    def row(id: str, text: str, **kw):
        row = {"id": id, "text": text, "source_id": "local",
               "license": "CC0-1.0", "rights_basis": "owned",
               "language": "en"}
        row.update(kw)
        return row

    def curate(self, dirname="first", **kwargs):
        return curate_dataset(self.input, self.policy, self.root / dirname, **kwargs)

    @staticmethod
    def read_rows(output):
        result = []
        for split in ("train", "validation", "test"):
            result.extend(json.loads(s) for s in
                          (output / f"{split}.jsonl").read_text().splitlines())
        return result

    def test_revocation_creates_immutable_valid_successor_and_lineage(self):
        self.rows(self.row("a", A), self.row("b", B))
        self.curate()
        parent = self.root / "first"
        parent_manifest = (parent / "manifest.json").read_bytes()
        tombstones = self.root / "tombstones.jsonl"
        tombstones.write_text('{"doc_id":"local:a"}\n', encoding="utf-8")
        child_manifest = revoke_snapshot(parent, tombstones, self.root / "second")
        child = self.root / "second"
        self.assertTrue(validate_output(child)["valid"])
        self.assertEqual(child_manifest["totals"]["accepted"], 1)
        self.assertEqual(child_manifest["lineage"]["removed_total"], 1)
        self.assertEqual(child_manifest["lineage"]["operation"], "revocation")
        self.assertEqual({r["id"] for r in self.read_rows(child)}, {"local:b"})
        self.assertEqual((parent / "manifest.json").read_bytes(), parent_manifest)
        receipt = (child / "revoked.jsonl").read_text()
        self.assertNotIn("local:a", receipt)
        self.assertNotIn(A[:40], receipt)
        self.assertNotIn("local:a", (child / "audit.jsonl").read_text())
        tombstones.write_text(json.dumps({
            "sha256": fingerprint(B)}) + "\n")
        next_generation = revoke_snapshot(child, tombstones, self.root / "third")
        self.assertEqual(next_generation["totals"]["accepted"], 0)
        self.assertEqual(self.read_rows(self.root / "third"), [])
        self.assertTrue(validate_output(self.root / "third")["valid"])

    def test_revoke_rejects_malformed_tombstones_without_publishing(self):
        self.rows(self.row("a", A))
        self.curate()
        denied = self.root / "tombstones.jsonl"
        for content in ('{"missing":"all"}\n', '{"sha256":"123"}\n', '{oops}\n'):
            denied.write_text(content)
            target = self.root / "never-produced"
            with self.assertRaises(ValueError):
                revoke_snapshot(self.root / "first", denied, target)
            self.assertFalse(target.exists())

    def test_revoke_fails_closed_on_parent_corruption(self):
        self.rows(self.row("a", A))
        self.curate()
        parent = self.root / "first"
        (parent / "train.jsonl").write_text("tampered\n")
        tombstones = self.root / "tombstones.jsonl"
        tombstones.write_text('{"doc_id":"local:a"}\n')
        with self.assertRaises(ValueError):
            revoke_snapshot(parent, tombstones, self.root / "child")
        self.assertFalse((self.root / "child").exists())

    def test_revocation_receipt_tampering_fails_verification(self):
        self.rows(self.row("a", A))
        self.curate()
        tombstones = self.root / "tombstones.jsonl"
        tombstones.write_text('{"doc_id":"local:a"}\n')
        revoke_snapshot(self.root / "first", tombstones, self.root / "second")
        path = self.root / "second" / "revoked.jsonl"
        path.write_text(path.read_text() + '{"decision":"removed"}\n')
        with self.assertRaisesRegex(ValueError, "corrupt revocation receipt"):
            validate_output(self.root / "second")

    def test_commercial_rights_fail_closed_on_legacy_and_unapproved_v2(self):
        self.rows(self.row("a", A))
        with self.assertRaisesRegex(ValueError, "schema v2"):
            self.curate(config=CurationConfig(commercial_use=True,
                                             as_of_date="2026-11-25"))
        self.policy.write_text(json.dumps({
            "schema_version": 2,
            "sources": [{
                "source_id": "local", "license": "CC0-1.0",
                "rights_basis": "owned", "training_allowed": True,
                "allowed_uses": ["pretraining"],
                "commercial_allowed": False,
                "redistribution_allowed": False,
                "retention_until": None,
                "review": {"reviewed_by": "synthetic-test-author",
                           "reviewed_at": "2026-11-24",
                           "evidence_sha256": "b" * 64}
            }]
        }))
        m = self.curate("commercial_denied", config=CurationConfig(
            commercial_use=True, as_of_date="2026-11-25"))
        self.assertEqual(m["totals"].get("accepted", 0), 0)
        self.assertEqual(m["totals"]["rejected_source_commercial_rights_not_granted"], 1)

    def test_revoke_rejects_existing_destination(self):
        self.rows(self.row("a", A))
        self.curate()
        tombstones = self.root / "tombstones.jsonl"
        tombstones.write_text('{"doc_id":"local:a"}\n')
        with self.assertRaises(FileExistsError):
            revoke_snapshot(self.root / "first", tombstones, self.root / "first")

    def test_quality_report_human_metrics_are_limited_to_accepted_subset(self):
        self.rows(self.row("a", A, domain="science"),
                  self.row("b", B, domain="code"))
        self.curate()
        parent = self.root / "first"
        data = self.read_rows(parent)
        self.assertEqual(len(data), 2)
        labels = self.root / "labels.jsonl"
        labels.write_text("".join(json.dumps(item) + "\n" for item in (
            {"sha256": data[0]["sha256"], "quality_label": 1},
            {"sha256": data[1]["sha256"], "quality_label": 0},
            {"sha256": "0"*64, "quality_label": 1},
        )))
        result = evaluate_quality(parent, labels_path=labels, threshold=0.0)
        scores = result["human_labels"]
        self.assertEqual(scores["matched"], 2)
        self.assertEqual(scores["unmatched"], 1)
        self.assertEqual((scores["tp"], scores["fp"]), (1, 1))
        self.assertEqual(scores["precision"], 0.5)
        self.assertEqual(scores["recall"], 1.0)
        self.assertIn("science", result["domain_counts"])
        self.assertIn("code", result["domain_counts"])
        self.assertEqual(result["scores_by_language"]["en"]["count"], 2)
        report = self.root / "quality.json"
        write_report(result, report)
        self.assertNotIn(A[:40], report.read_text())
        with self.assertRaises(FileExistsError):
            write_report(result, report)

    def test_empty_snapshot_quality_report(self):
        self.rows(self.row("a", "tiny"))
        self.curate()
        result = evaluate_quality(self.root / "first")
        self.assertEqual(result["human_labels"]["matched"], 0)
        self.assertIsNone(result["human_labels"]["precision"])
        self.assertEqual(result["scores_by_split"]["train"]["count"], 0)

    def test_rights_v2_requires_review_and_fixed_run_date(self):
        self.policy.write_text(json.dumps({
            "schema_version": 2, "sources": [{
                "source_id": "local", "license": "CC0-1.0",
                "rights_basis": "owned", "training_allowed": True,
                "allowed_uses": ["pretraining"],
                "commercial_allowed": False,
                "redistribution_allowed": False,
                "retention_until": "2026-11-25",
                "review": {
                    "reviewed_by": "synthetic-test-operator",
                    "reviewed_at": "2026-11-24",
                    "evidence_sha256": "a" * 64
                }
            }]
        }))
        self.rows(self.row("a", A))
        with self.assertRaisesRegex(ValueError, "as-of-date"):
            self.curate()
        manifest = self.curate("allowed", config=CurationConfig(as_of_date="2026-11-25"))
        self.assertEqual(manifest["totals"]["accepted"], 1)
        self.assertEqual(manifest["configuration"]["as_of_date"], "2026-11-25")
        expired = self.curate("expired", config=CurationConfig(as_of_date="2026-11-26"))
        self.assertEqual(expired["totals"]["rejected_source_rights_not_current"], 1)
        self.assertEqual(expired["totals"].get("accepted", 0), 0)

    def test_v2_missing_attestation_rejected(self):
        self.policy.write_text(json.dumps({
            "schema_version": 2, "sources": [{
                "source_id": "local", "license": "CC0-1.0",
                "rights_basis": "owned", "training_allowed": True
            }]
        }))
        with self.assertRaisesRegex(ValueError, "review evidence"):
            load_policy(self.policy)

    def test_script_sanity_is_opt_in_and_does_not_claim_full_lang_id(self):
        tamil = ("தமிழ் அறிவியல் ஆராய்ச்சி மாணவர்கள் கற்றல் "
                 "முறைகள் ஆய்வு விளக்கம் பல்கலைக்கழகம் ") * 8
        good, share = script_consistent(tamil, "ta")
        self.assertTrue(good)
        self.assertGreater(share["tamil"], 0.65)
        self.assertFalse(script_consistent(A, "ta")[0])
        self.assertTrue(script_consistent(A, "en")[0])
        self.rows(self.row("wrong", A, language="ta"),
                  self.row("right", tamil, language="ta"))
        m = self.curate(config=CurationConfig(script_sanity=True, min_quality=0.0))
        self.assertEqual(m["totals"]["rejected_script_mismatch"], 1)
        self.assertEqual(m["totals"]["accepted"], 1)
        self.assertEqual(self.read_rows(self.root / "first")[0]["language_declared"], "ta")

    def test_broken_rights_metadata_is_not_silently_accepted(self):
        self.rows(self.row("a", A), self.row("b", B, domain="untrusted-tag"))
        m = self.curate()
        self.assertEqual(m["totals"]["accepted"], 2)
        domains = {r["domain_declared"] for r in self.read_rows(self.root / "first")}
        self.assertEqual(domains, {"unlabeled"})

    def test_cli_assessment_and_revocation(self):
        self.rows(self.row("a", A))
        self.curate()
        self.assertEqual(main(["assess", "--snapshot", str(self.root / "first"),
                               "--report", str(self.root / "report.json")]), 0)
        tombstones = self.root / "tombstones.jsonl"
        tombstones.write_text('{"doc_id":"local:a"}\n')
        self.assertEqual(main([
            "revoke", "--parent", str(self.root / "first"),
            "--tombstones", str(tombstones), "--output", str(self.root / "second")
        ]), 0)
        self.assertEqual(main([
            "verify", "--output", str(self.root / "second")
        ]), 0)


if __name__ == "__main__":
    unittest.main()
