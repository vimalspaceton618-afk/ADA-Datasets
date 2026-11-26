"""Train a real local fastText model from a small original multilingual fixture."""
from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from ada_datasets.quality_training import _calibrate, train_quality_model
from ada_datasets.release_intake import FastTextQuality


_LABELS = {
    "en": (
        "Reproducible evidence careful measurements scientific analysis",
        "Offers coupon coupon subscribe discount clutter repeated",
    ),
    "ta": (
        "ஆராய்ச்சி அளவீடு நம்பகமான ஆதாரம் தெளிவான விளக்கம்",
        "விளம்பரம் விற்பனை சலுகை வாங்குங்கள் சலுகை",
    ),
    "hi": (
        "परीक्षण परिणाम स्पष्ट व्याख्या विश्वसनीय प्रमाण विवरण",
        "विज्ञापन खरीदें छूट विज्ञापन सौदा सदस्यता",
    ),
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_labels(destination, partition, shared_family=None):
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for lang, texts in _LABELS.items():
            for index, text in enumerate(texts):
                row = {
                    "text": f"{text} {partition} note __label__example </s>",
                    "family_id": shared_family if index == 0 and lang == "en" else f"{partition}-{lang}-{index}",
                    "language": lang, "should_keep": index == 0,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class CalibrationTest(unittest.TestCase):
    def test_threshold_uses_higher_precision_when_f1_ties(self):
        score, metrics = _calibrate([(0.8, True), (0.7, False), (0.65, False), (0.5, True)])
        self.assertEqual(score, 0.8)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)


@unittest.skipUnless(importlib.util.find_spec("fasttext"), "quality-model extras required")
class QualityTrainingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.fit = self.root / "fit.jsonl"
        self.calibration = self.root / "calibration.jsonl"
        write_labels(self.fit, "fit", shared_family="shared")
        write_labels(self.calibration, "calibration", shared_family="independent")

    def train(self, output, **extra):
        kwargs = {
            "fit_sha256": digest(self.fit), "calibration_sha256": digest(self.calibration),
            "model_license": "CC0-1.0", "min_calibration_per_class": 1, **extra,
        }
        return train_quality_model(self.fit, self.calibration, output, **kwargs)

    def test_real_model_reproducibly_trains_calibrates_and_loads(self):
        first, second = self.root / "first", self.root / "second"
        report = self.train(first)
        self.assertEqual(self.train(second), report)
        self.assertEqual(digest(first / "quality.bin"), digest(second / "quality.bin"))
        self.assertEqual(report["parameters"]["thread"], 1)
        self.assertEqual(report["fit"]["sha256"], digest(self.fit))
        self.assertEqual(report["calibration"]["sha256"], digest(self.calibration))
        self.assertEqual(set(report["model"]["calibration"]), set(_LABELS))
        model = FastTextQuality(
            first / "quality.bin", expected_sha256=report["model"]["sha256"],
            declared_license=report["model"]["license"],
        )
        self.assertTrue(0.0 <= model.keep_probability(_LABELS["en"][0]) <= 1.0)
        report_text = (first / "training-report.json").read_text(encoding="utf-8")
        self.assertNotIn(_LABELS["en"][0], report_text)
        self.assertEqual(
            {path.name for path in first.iterdir()},
            {"training-report.json", "quality-model.json", "quality.bin"},
        )

    def test_bad_pin_and_fit_calibration_family_overlap_publish_nothing(self):
        output = self.root / "output"
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.train(output, fit_sha256="0" * 64)
        self.assertFalse(output.exists())
        write_labels(self.calibration, "calibration", shared_family="shared")
        with self.assertRaisesRegex(ValueError, "fit family"):
            self.train(output)
        self.assertFalse(output.exists())
