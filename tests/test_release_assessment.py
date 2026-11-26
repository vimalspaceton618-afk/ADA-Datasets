"""Offline tests for aggregate evaluations of release intake gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ada_datasets.release_assessment import (
    add_assessment_parser,
    assess_release_cli,
    evaluate_release_intake,
)


class StubQualityModel:
    def metadata(self):
        return {
            "family": "fasttext-quality",
            "model_sha256": "a" * 64,
            "license_declared": "CC0-1.0",
            "path": "/private/model.bin",
        }

    def keep_probability(self, _text):
        return 0.6


class ReleaseAssessmentTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.gold = self.root / "gold.jsonl"

    def write_gold(self, rows):
        self.gold.write_bytes(
            "".join(json.dumps(row, ensure_ascii=False) + "\r\n" for row in rows)
            .encode("utf-8")
        )

    def test_aggregate_confusion_uses_exact_bytes_and_omits_private_text(self):
        self.write_gold([
            {"text": "Contact reviewer@example.invalid", "language": "en",
             "contains_pii": True, "should_keep": False},
            {"text": "தமிழ் மாதிரி உரை", "language": "ta",
             "contains_pii": False, "should_keep": True},
            {"text": "मानव समीक्षा", "language": "hi", "should_keep": False},
        ])
        report = evaluate_release_intake(self.gold, min_quality=0)
        self.assertEqual(report["gold_file_sha256"], hashlib.sha256(self.gold.read_bytes()).hexdigest())
        self.assertEqual(report["rows"], 3)
        self.assertEqual(report["privacy"]["labeled_rows"], 2)
        self.assertEqual(report["privacy"]["metrics"]["tp"], 1)
        self.assertEqual(report["privacy"]["metrics_by_language"]["ta"]["tn"], 1)
        self.assertEqual(report["quality"]["metrics"]["tp"], 1)
        self.assertEqual(report["quality"]["metrics"]["fp"], 2)
        self.assertNotIn("reviewer@example.invalid", json.dumps(report))
        self.assertNotIn("मानव समीक्षा", json.dumps(report, ensure_ascii=False))

    def test_model_thresholds_are_applied_by_language(self):
        self.write_gold([
            {"text": "An English quality benchmark", "language": "en", "should_keep": True},
            {"text": "தமிழ் மாதிரி உரை", "language": "ta", "should_keep": True},
        ])
        model = StubQualityModel()
        with self.assertRaisesRegex(ValueError, "calibration"):
            evaluate_release_intake(self.gold, quality_model=model, quality_calibration={"en": 0.6})
        report = evaluate_release_intake(
            self.gold, quality_model=model,
            quality_calibration={"en": 0.6, "ta": 0.8, "hi": 0.7},
        )
        self.assertEqual(report["quality"]["metrics_by_language"]["en"]["tp"], 1)
        self.assertEqual(report["quality"]["metrics_by_language"]["ta"]["fn"], 1)
        self.assertEqual(report["quality_model"]["model_sha256"], "a" * 64)
        self.assertNotIn("/private/model.bin", json.dumps(report))

    def test_sample_caps_and_strict_labels(self):
        sample = {"text": "An English evaluation sample", "language": "en",
                  "contains_pii": False}
        self.write_gold([sample, sample])
        with self.assertRaisesRegex(ValueError, "size cap"):
            evaluate_release_intake(self.gold, max_rows=1)
        self.write_gold([dict(sample, contains_pii=1)])
        with self.assertRaisesRegex(ValueError, "invalid labeled gold sample"):
            evaluate_release_intake(self.gold)
        self.gold.write_bytes(b"\xffraw-private-diagnostic")
        with self.assertRaisesRegex(ValueError, "^invalid gold sample JSON on line 1$"):
            evaluate_release_intake(self.gold)

    def test_cli_returns_report_and_refuses_overwrite(self):
        self.write_gold([
            {"text": "An English evaluation sample", "language": "en", "should_keep": True}
        ])
        parser = argparse.ArgumentParser()
        add_assessment_parser(parser.add_subparsers(dest="command", required=True))
        output = self.root / "report.json"
        command = ["assess-release", "--gold", str(self.gold), "--report", str(output)]
        args = parser.parse_args(command)
        self.assertEqual(assess_release_cli(args), json.loads(output.read_text()))
        with self.assertRaises(FileExistsError):
            assess_release_cli(args)
        args = parser.parse_args(command + ["--quality-model", "local.bin"])
        with self.assertRaisesRegex(ValueError, "supplied together"):
            assess_release_cli(args)


if __name__ == "__main__":
    unittest.main()
