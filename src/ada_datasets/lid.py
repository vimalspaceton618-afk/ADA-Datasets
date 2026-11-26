"""Opt-in local fastText lid.176 language identifier with artifact pinning.

No automatic model downloads. The older fastText lid.176.bin/.ftz family is
distributed separately under CC-BY-SA-3.0. It must NOT be confused with the
HF-hosted NLLB fasttext-language-identification model (CC-BY-NC-4.0).
License declarations here are operator assertions; legal review is external.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SUPPORTED = ("en", "ta", "hi")
_MODEL_FAMILY = "fasttext-lid176"


@dataclass(frozen=True)
class LanguagePrediction:
    language: str
    confidence: float

    def __post_init__(self) -> None:
        if (not isinstance(self.language, str) or
                not re.fullmatch(r"[a-z]{2,3}", self.language) or
                isinstance(self.confidence, bool) or
                not isinstance(self.confidence, (float, int)) or
                not math.isfinite(self.confidence) or
                not 0 <= self.confidence <= 1):
            raise ValueError("invalid language-ID prediction")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FastTextLID176:
    """Load exactly a local, operator-reviewed original fastText lid.176 model."""

    def __init__(self, model_path: Path, *, expected_sha256: str,
                 declared_license: str):
        path = Path(model_path)
        if declared_license != "CC-BY-SA-3.0":
            raise ValueError(
                "This adapter only supports the original CC-BY-SA-3.0 lid.176 "
                "family; the HF-hosted NLLB CC-BY-NC-4.0 model is different."
            )
        if not _DIGEST.fullmatch(expected_sha256):
            raise ValueError("model SHA-256 must be 64 lowercase hexadecimal digits")
        if path.suffix not in (".bin", ".ftz") or not path.is_file() or path.is_symlink():
            raise ValueError("supply a local regular .bin/.ftz fastText model file")
        observed = _sha256_file(path)
        if observed != expected_sha256:
            raise ValueError("language-ID model SHA-256 mismatch; refusing to load")
        try:
            import fasttext  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "optional dependency missing: pip install fasttext"
            ) from exc
        self._model = fasttext.load_model(str(path))
        if _sha256_file(path) != observed:
            raise ValueError("language-ID model changed while loading")
        self.model_sha256 = observed
        self.license_declared = declared_license

    def predict(self, text: str) -> LanguagePrediction:
        if not isinstance(text, str):
            raise ValueError("language-ID input must be text")
        # fastText requires a single UTF-8 line. Bounded input prevents
        # accidental unbounded model work on a multi-MB document.
        cleaned = " ".join(text.split())[:50_000]
        if not cleaned:
            raise ValueError("language-ID input is empty")
        labels, scores = self._model.predict(cleaned, k=1)
        if len(labels) != 1 or len(scores) != 1 or not isinstance(labels[0], str):
            raise ValueError("fastText returned an invalid prediction")
        if not re.fullmatch(r"__label__[a-z]{2,3}", labels[0]):
            raise ValueError("unexpected language label: expected original lid.176 codes")
        confidence = float(scores[0])
        # fastText can add a small epsilon at prediction time.
        if 1.0 < confidence <= 1.0001:
            confidence = 1.0
        return LanguagePrediction(labels[0][9:], confidence)

    def metadata(self) -> dict:
        return {
            "family": _MODEL_FAMILY,
            "model_sha256": self.model_sha256,
            "license_declared": self.license_declared,
            "source": "https://fasttext.cc/docs/en/language-identification.html",
            "model_bundled": False,
            "license_verified_automatically": False,
        }


def accepted_prediction(prediction: LanguagePrediction, declared: str,
                        threshold: float) -> str | None:
    """Return a content-free rejection code, or None for a matching result."""
    if not 0 < threshold <= 1:
        raise ValueError("language-ID threshold must be in (0,1]")
    if not isinstance(prediction, LanguagePrediction):
        raise ValueError("language-ID provider returned the wrong result type")
    if prediction.confidence < threshold:
        return "language_id_low_confidence"
    if prediction.language != declared or declared not in _SUPPORTED:
        return "language_id_mismatch"
    return None
