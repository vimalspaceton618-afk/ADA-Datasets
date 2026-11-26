"""Boundary tests for byte-exact intake and release gates."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ada_datasets.core import CurationConfig, curate_dataset, fingerprint
from ada_datasets.evaluation import evaluate_filter
from ada_datasets.hf_adapter import adapt_hf
from ada_datasets.release_intake import prepare_inputs, release_has_pii


TEXT = "Careful scientific observers document precision and report uncertainty transparently."
BENCHMARK = (
    "Independent auditors measure every selected outcome using a repeatable procedure "
    "whose assumptions and estimation limits appear in the published appendix."
)


def encode(row, newline=b"\n"):
    return json.dumps(row, ensure_ascii=False).encode("utf-8") + newline


class IntakeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "raw.jsonl"
        self.policy = self.root / "policy.json"
        self.output = self.root / "output"
        self.policy_row = {
            "source_id": "owned", "license": "CC0-1.0", "rights_basis": "owned",
            "training_allowed": True, "commercial_allowed": True,
            "redistribution_allowed": False, "retention_until": None,
            "allowed_uses": ["pretraining"],
            "review": {
                "reviewed_by": "fixture reviewer", "reviewed_at": "2026-01-01",
                "evidence_sha256": "1" * 64,
            },
            "attribution": "Unit test fixture, version one",
        }
        self.write_policy()
        self.recipe = {
            "release_use": "internal",
            "curation": {
                "as_of_date": "2026-11-25", "min_quality": 0, "min_chars": 8, "min_words": 3,
            },
        }

    def write_policy(self, version=2):
        self.policy.write_bytes(encode({"schema_version": version, "sources": [self.policy_row]}))

    def row(self, identifier, **changes):
        return {
            "id": identifier, "source_id": "owned", "language": "en", "text": TEXT,
            "license": "CC0-1.0", "rights_basis": "owned", **changes,
        }

    def run_intake(self, holdout=None, denylist=None, expected_sha=None):
        return prepare_inputs(
            self.raw, self.output, recipe=self.recipe, policy_path=self.policy,
            holdout_path=holdout, denylist_path=denylist,
            input_sha256=expected_sha or sha256(self.raw.read_bytes()).hexdigest(),
        )

    def rows(self, basename):
        return [
            json.loads(line)
            for line in (self.output / basename).read_bytes().splitlines()
        ]

    def test_byte_exact_hashes_in_legacy_evaluator_and_release(self):
        lines = [encode(self.row("kept"), b"\r\n"), b"{\xff}\r\n", b'{"not":"a source"}']
        self.raw.write_bytes(b"".join(lines))
        self.assertEqual(self.run_intake()["candidates"], 1)
        audit = self.rows("audit.jsonl")
        self.assertEqual(audit[0]["reason"], "invalid_utf8")
        self.assertEqual([row["input_line_sha256"] for row in audit],
                         [sha256(line).hexdigest() for line in lines[1:]])
        self.assertEqual(
            self.rows("candidates.jsonl")[0]["input_line_sha256"], sha256(lines[0]).hexdigest()
        )
        snapshot = self.root / "legacy"
        curate_dataset(self.raw, self.policy, snapshot, config=CurationConfig(**self.recipe["curation"]))
        labels = self.root / "labels.jsonl"
        labels.write_bytes(b"".join(
            encode({
                "line": num, "input_line_sha256": sha256(line).hexdigest(),
                "should_keep": num == 1,
            })
            for num, line in enumerate(lines, 1)
        ))
        report = evaluate_filter(snapshot, self.raw, labels)
        self.assertEqual(report["metrics"]["tp"], 1)
        self.assertEqual(report["metrics"]["tn"], 2)

    def test_holdout_passage_catches_embedding_and_exact_records(self):
        holdout = self.root / "holdout.jsonl"
        holdout.write_bytes(encode({"text": BENCHMARK}))
        self.raw.write_bytes(b"".join([
            encode(self.row("exact", text=BENCHMARK)),
            encode(self.row("excerpt", text=f"{TEXT} {BENCHMARK.upper()} {TEXT} {TEXT}")),
            encode(self.row("clean")),
        ]))
        counts = self.run_intake(holdout)
        self.assertEqual(counts["rejection_reasons"], {
            "benchmark_contamination": 1, "benchmark_passage_contamination": 1,
        })
        self.assertEqual(self.rows("candidates.jsonl")[0]["id"], "owned:clean")

    def test_near_holdout_matching_with_no_thirteen_word_passage(self):
        holdout = self.root / "holdout.jsonl"
        short_text = "Audit reviewers examine reproducible summaries with documented uncertainty and sources."
        holdout.write_bytes(encode({"text": short_text}))
        self.recipe["curation"]["min_near_jaccard"] = 0.65
        self.raw.write_bytes(encode(self.row("near", text=short_text.replace("sources.", "reviews."))))
        self.assertEqual(self.run_intake(holdout)["rejection_reasons"], {"benchmark_contamination": 1})

    def test_optout_ledger_does_not_trust_unapproved_records(self):
        self.raw.write_bytes(b"".join([
            encode(self.row("approved", opt_out=True)),
            encode(self.row("mismatched", opt_out=True, license="CC-BY-4.0")),
            encode(self.row("unapproved", opt_out=True, source_id="other")),
            encode(self.row("metadata-only", text=None, do_not_train=True)),
        ]))
        counts = self.run_intake()
        self.assertEqual(counts["rejected"], 4)
        self.assertEqual(self.rows("suppression.jsonl"), [
            {"doc_id_sha256": sha256(b"owned:approved").hexdigest(), "sha256": fingerprint(TEXT)},
            {"doc_id_sha256": sha256(b"owned:metadata-only").hexdigest()},
        ])
        self.assertIsNone(self.rows("audit.jsonl")[2]["source_id"])

    def test_distribution_requires_current_reviewed_source_and_approved_attribution(self):
        self.raw.write_bytes(encode(self.row("a")))
        self.recipe["release_use"] = "redistribution"
        counts = self.run_intake()
        self.assertEqual(counts["rejection_reasons"], {"source_redistribution_rights_not_granted": 1})
        self.output = self.root / "no-attribution"
        self.policy_row.pop("attribution")
        self.write_policy()
        with self.assertRaisesRegex(ValueError, "attribution"):
            self.run_intake()
        self.assertFalse(self.output.exists())

    def test_id_tombstone_records_approved_alias_content(self):
        denylist = self.root / "denylist.jsonl"
        denylist.write_bytes(encode({"doc_id": "owned:removed"}))
        self.raw.write_bytes(
            encode(self.row("removed", text="A forged fingerprint should not appear.", license="other"))
            + encode(self.row("removed"))
        )
        counts = self.run_intake(denylist=denylist)
        self.assertEqual(counts["rejection_reasons"], {"denylisted": 1, "provenance_mismatch": 1})
        self.assertEqual(self.rows("suppression.jsonl"), [
            {"doc_id_sha256": sha256(b"owned:removed").hexdigest(), "sha256": fingerprint(TEXT)},
        ])

    def test_duplicate_json_consent_and_multilingual_privacy(self):
        text_with_private_number = "स्थानीय कार्यालय का संपर्क ९८७६५ ४३२१० इतिहास अभिलेख में है।"
        self.assertTrue(release_has_pii(text_with_private_number))
        self.assertTrue(release_has_pii("A question labeled ָ from test@example.invalid"))
        self.raw.write_bytes(b"".join([
            encode(self.row("unsafe", text=text_with_private_number, language="hi")),
            b'{"id":"ambiguous","opt_out":true,"opt_out":false}\n',
        ]))
        self.assertEqual(self.run_intake()["rejection_reasons"],
                         {"invalid_json_or_metadata": 1, "pii_or_secret": 1})
        self.assertNotIn("९८७६५", (self.output / "audit.jsonl").read_text(encoding="utf-8"))

    def test_pinned_fasttext_quality_thresholds(self):
        self.raw.write_bytes(encode(self.row("accepted")) + encode(self.row("rejected", language="ta")))
        model = self.root / "quality.bin"
        model.write_bytes(b"unit-test mock model")
        self.recipe["quality_model"] = {
            "path": str(model), "sha256": sha256(model.read_bytes()).hexdigest(),
            "license": "operator fixture license", "calibration": {"en": 0.7, "ta": 0.9, "hi": 0.8},
        }
        fake_model = SimpleNamespace(
            get_labels=lambda: ["__label__keep", "__label__drop"],
            predict=lambda text, k: (["__label__keep", "__label__drop"], [0.8, 0.2]),
        )
        with patch.dict(sys.modules, {"fasttext": SimpleNamespace(load_model=lambda _: fake_model)}):
            self.assertEqual(self.run_intake()["rejection_reasons"], {"low_quality": 1})
        self.assertEqual(self.rows("candidates.jsonl")[0]["quality_score"], 0.8)

    def test_pin_error_never_publishes(self):
        self.raw.write_bytes(encode(self.row("a")))
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_intake(expected_sha="0" * 64)
        self.assertFalse(self.output.exists())

    def test_hf_stream_caps_seen_rows_and_does_not_pull_after_success(self):
        self.policy_row.update(
            hf_dataset_id="fixture/corpus", hf_config="default", hf_split="train",
            hf_revision="a" * 40,
        )
        self.write_policy(version=1)
        args = SimpleNamespace(
            acknowledge_source_rights=True, max_rows=2, revision="a" * 40,
            output=self.root / "adapter.jsonl", sources=self.policy, source_id="owned",
            dataset_id="fixture/corpus", config="default", split="train", language="en",
        )
        pulls = Counter()

        def stream(eligible):
            for index in itertools.count():
                pulls[eligible] += 1
                yield {"text": TEXT, "id": str(index)} if eligible else {"unusable": True}

        for eligible, expected, seen in ((True, 2, 2), (False, 0, 20)):
            args.output = self.root / f"adapter-{eligible}.jsonl"
            dataset = SimpleNamespace(load_dataset=lambda *_, **__: stream(eligible))
            with patch.dict(sys.modules, {"datasets": dataset}):
                summary = adapt_hf(args)
            self.assertEqual(summary["exported_rows"], expected)
            self.assertEqual(summary["scanned_rows"], seen)
            self.assertEqual(pulls[eligible], seen)
