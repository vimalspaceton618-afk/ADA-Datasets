"""v0.3: independently evaluated language identification and full-input filtering."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ada_datasets.cli import main
from ada_datasets.core import CurationConfig, curate_dataset, file_digest
from ada_datasets.evaluation import (
    evaluate_filter, evaluate_language_id, _wilson,
)
from ada_datasets.lid import (
    FastTextLID176, LanguagePrediction, accepted_prediction,
)

SCIENCE = (
    "An experiment in physics begins with a stated research question and a "
    "repeatable scientific protocol. Observers calibrate instruments, record "
    "measurements, monitor uncertainty and compare results against reference "
    "observations. They document numerical assumptions in detail and allow "
    "other researchers to reproduce and challenge the underlying evidence."
)
CODE = (
    "Software architecture benefits from strongly specified interfaces and "
    "well designed regression tests. Developers isolate input validation, "
    "observe expected behavior, measure fault recovery and record performance "
    "boundaries. High quality code reviews focus on correctness and security "
    "rather than merely reporting that a program ran once without an error."
)
MATH = (
    "Mathematical reasoning relies on explicitly stated hypotheses and "
    "carefully justified deductions. Learners compare alternative derivations, "
    "test limiting cases and inspect numerical results. They preserve a "
    "complete argument for independent peer review while distinguishing "
    "empirical approximations from rigorously established mathematical claims."
)


class DummyClassifier:
    """Deterministic mock inference; never claim that it tests real fastText."""
    def metadata(self):
        return {
            "family": "fasttext-lid176",
            "model_sha256": "a" * 64,
            "license_declared": "CC-BY-SA-3.0",
            "model_bundled": False,
        }

    def predict(self, text):
        if "science" in text:
            return LanguagePrediction("en", 0.99)
        if "Software" in text:
            return LanguagePrediction("ta", 0.92)
        return LanguagePrediction("en", 0.30)


class V03EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "raw.jsonl"
        self.policy = self.root / "source-policy.json"
        self.policy.write_text(json.dumps({
            "schema_version": 1,
            "sources": [
                {"source_id": "owned", "license": "CC0-1.0",
                 "rights_basis": "owned", "training_allowed": True},
                {"source_id": "blocked", "license": "CC0-1.0",
                 "rights_basis": "owned", "training_allowed": False},
            ]
        }))

    @staticmethod
    def row(id, text, **extra):
        result = {
            "id": id, "text": text, "source_id": "owned",
            "rights_basis": "owned", "license": "CC0-1.0",
            "language": "en",
        }
        result.update(extra)
        return result

    def write_rows(self, rows):
        self.input.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows), encoding="utf-8",
        )

    def curate(self, *, config=None, language_identifier=None):
        return curate_dataset(
            self.input, self.policy, self.root / "snapshot",
            config=config, language_identifier=language_identifier,
        )

    def test_real_adapter_requires_pinned_local_weight_and_correct_license(self):
        p = self.root / "lid.176.ftz"
        p.write_bytes(b"synthetic-test-model-no-real-weights")
        digest = file_digest(p)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            FastTextLID176(
                p, expected_sha256="b" * 64,
                declared_license="CC-BY-SA-3.0",
            )
        with self.assertRaisesRegex(ValueError, "CC-BY-SA-3.0"):
            FastTextLID176(
                p, expected_sha256=digest, declared_license="CC-BY-NC-4.0",
            )

        fake_engine = types.SimpleNamespace(
            predict=lambda text, k: (("__label__ta",), (0.91,))
        )
        fake_module = types.SimpleNamespace(load_model=lambda path: fake_engine)
        with patch.dict(sys.modules, {"fasttext": fake_module}):
            loaded = FastTextLID176(
                p, expected_sha256=digest,
                declared_license="CC-BY-SA-3.0",
            )
        self.assertEqual(loaded.predict("தமிழ் உரை").language, "ta")
        self.assertAlmostEqual(loaded.predict("தமிழ் உரை").confidence, 0.91)
        self.assertEqual(loaded.metadata()["model_sha256"], digest)
        self.assertFalse(loaded.metadata()["model_bundled"])

    def test_language_id_prediction_validation(self):
        with self.assertRaises(ValueError):
            LanguagePrediction("en", float("nan"))
        with self.assertRaises(ValueError):
            LanguagePrediction("eng_Latn", 0.80)
        self.assertEqual(
            accepted_prediction(LanguagePrediction("ta", 0.91), "en", 0.8),
            "language_id_mismatch",
        )
        self.assertEqual(
            accepted_prediction(LanguagePrediction("en", 0.40), "en", 0.8),
            "language_id_low_confidence",
        )
        self.assertIsNone(
            accepted_prediction(LanguagePrediction("en", 0.99), "en", 0.8)
        )

    def test_language_id_gate_on_real_curator_code(self):
        self.write_rows([
            self.row("matched", SCIENCE),
            self.row("mismatch", CODE),
            self.row("abstain", MATH),
        ])
        manifest = self.curate(language_identifier=DummyClassifier())
        self.assertEqual(manifest["totals"]["accepted"], 1)
        self.assertEqual(manifest["totals"]["rejected_language_id_mismatch"], 1)
        self.assertEqual(
            manifest["totals"]["rejected_language_id_low_confidence"], 1
        )
        self.assertEqual(
            manifest["language_model"]["model_sha256"], "a" * 64
        )
        self.assertEqual(
            manifest["configuration"]["lid_min_confidence"], 0.8
        )

    def test_unpinned_identifier_is_rejected_before_any_output(self):
        class UnknownModel(DummyClassifier):
            def metadata(self):
                return {"name": "unverified-model"}

        self.write_rows([self.row("x", SCIENCE)])
        with self.assertRaisesRegex(ValueError, "pinned licensed model"):
            self.curate(language_identifier=UnknownModel())
        self.assertFalse((self.root / "snapshot").exists())

    def test_lid_gold_evaluation_reports_confusion_abstention_without_text(self):
        gold = self.root / "gold-language.jsonl"
        samples = [
            {"text": SCIENCE, "language": "en"},
            {"text": CODE, "language": "ta"},
            {"text": MATH, "language": "hi"},
            {"text": CODE, "language": "fr"},
        ]
        gold.write_text("".join(json.dumps(s) + "\n" for s in samples))
        report = evaluate_language_id(
            DummyClassifier(), gold, min_confidence=0.80
        )
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["by_gold"]["en"]["support"], 1)
        self.assertEqual(report["by_gold"]["hi"]["abstained"], 1)
        self.assertEqual(report["by_gold"]["ta"]["recall_with_abstentions"], 1.0)
        self.assertEqual(report["confusion_by_gold"]["other"]["ta"], 1)
        self.assertEqual(report["coverage"], 0.75)
        self.assertNotIn(SCIENCE[:25], json.dumps(report))
        self.assertNotIn(CODE[:25], json.dumps(report))

    def test_lid_gold_validation_rejects_unlabeled_or_malformed(self):
        gold = self.root / "lid-gold.jsonl"
        gold.write_text(json.dumps({"text": SCIENCE}) + "\n")
        with self.assertRaisesRegex(ValueError, "gold row"):
            evaluate_language_id(DummyClassifier(), gold)
        gold.write_text(json.dumps({"text": SCIENCE, "language": "en"}) + "\n")
        with self.assertRaisesRegex(ValueError, "cap"):
            evaluate_language_id(DummyClassifier(), gold, max_rows=0)

    def test_filter_evaluation_covers_accepted_and_rejected_input_lines(self):
        rows = [
            self.row("good", SCIENCE),                  # TP
            self.row("blocked", CODE, source_id="blocked"), # TN
            self.row("undesirable-but-passes", CODE),   # FP
            self.row("desirable-short", "tiny"),         # FN
        ]
        self.write_rows(rows)
        manifest = self.curate()
        self.assertEqual(manifest["totals"]["lines"], 4)
        gold = self.root / "human-input-labels.jsonl"
        raw_lines = self.input.read_text().splitlines(keepends=True)
        labels = []
        for n, should_keep in enumerate((True, False, False, True), 1):
            labels.append({
                "line": n,
                "input_line_sha256": sha256(
                    raw_lines[n-1].encode("utf-8")
                ).hexdigest(),
                "should_keep": should_keep,
                "language": "en",
            })
        gold.write_text("".join(json.dumps(x) + "\n" for x in labels))
        report = evaluate_filter(self.root / "snapshot", self.input, gold)
        self.assertEqual(report["human_labels"], 4)
        self.assertEqual(report["input_lines"], 4)
        self.assertEqual(report["labeled_fraction"], 1.0)
        self.assertEqual(
            [report["metrics"][name] for name in ("tp", "fp", "tn", "fn")],
            [1, 1, 1, 1],
        )
        self.assertEqual(report["metrics"]["precision"], 0.5)
        self.assertEqual(report["metrics"]["recall"], 0.5)
        self.assertEqual(report["metrics_by_labeled_language"]["en"]["tp"], 1)
        self.assertIsNotNone(report["metrics"]["recall_wilson95"])
        self.assertNotIn(SCIENCE[:25], json.dumps(report))

        # Verify the new CLI subcommand is a genuine executable workflow.
        result_path = self.root / "filter-report.json"
        self.assertEqual(main([
            "evaluate-filter", "--snapshot", str(self.root / "snapshot"),
            "--input", str(self.input), "--gold", str(gold),
            "--report", str(result_path),
        ]), 0)
        self.assertTrue(result_path.exists())

    def test_filter_gold_digest_fails_closed_on_tampering(self):
        self.write_rows([self.row("good", SCIENCE)])
        self.curate()
        line = self.input.read_text()
        gold = self.root / "gold-filter.jsonl"
        gold.write_text(json.dumps({
            "line": 1, "input_line_sha256": "f" * 64,
            "should_keep": True,
        }) + "\n")
        with self.assertRaisesRegex(ValueError, "human label input hash mismatch"):
            evaluate_filter(self.root / "snapshot", self.input, gold)
        gold.write_text(json.dumps({
            "line": 1, "input_line_sha256": sha256(line.encode()).hexdigest(),
            "should_keep": True,
        }) + "\n")
        self.input.write_text("tampered raw input\n")
        with self.assertRaisesRegex(ValueError, "exact raw input"):
            evaluate_filter(self.root / "snapshot", self.input, gold)

    def test_filter_rejects_duplicates_and_boolean_coercion(self):
        self.write_rows([self.row("good", SCIENCE)])
        self.curate()
        line_hash = sha256(self.input.read_bytes()).hexdigest()
        gold = self.root / "gold.jsonl"
        item = {"line": 1, "input_line_sha256": line_hash,
                "should_keep": 1}
        gold.write_text(json.dumps(item) + "\n")
        with self.assertRaisesRegex(ValueError, "invalid human label"):
            evaluate_filter(self.root / "snapshot", self.input, gold)
        item["should_keep"] = True
        gold.write_text((json.dumps(item) + "\n") * 2)
        with self.assertRaisesRegex(ValueError, "invalid human label"):
            evaluate_filter(self.root / "snapshot", self.input, gold)

    def test_wilson_intervals_have_defined_zero_behavior(self):
        self.assertIsNone(_wilson(0, 0))
        lo, hi = _wilson(1, 1)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)

    def test_cli_refuses_missing_lid_license_before_running(self):
        self.write_rows([self.row("good", SCIENCE)])
        self.assertEqual(main([
            "run", "--input", str(self.input),
            "--sources", str(self.policy),
            "--output", str(self.root / "nothing"),
            "--lid-model", str(self.root / "lid.176.ftz"),
        ]), 2)
        self.assertFalse((self.root / "nothing").exists())


if __name__ == "__main__":
    unittest.main()
