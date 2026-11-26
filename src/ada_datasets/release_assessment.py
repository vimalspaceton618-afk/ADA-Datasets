"""Measure release privacy and quality decisions against private human labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .core import VERSION, CurationConfig, canonical_text, score_text
from .evaluation import _confusion_metrics
from .quality import write_report
from .release_intake import FastTextQuality, release_has_pii

_LANGUAGES = CurationConfig().supported_languages
_MAX_ROW_BYTES = 8 * 1024 * 1024
_MAX_GOLD_BYTES = 512 * 1024 * 1024
_MAX_GOLD_ROWS = 100_000


def _threshold(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1


def _outcome(expected: bool, predicted: bool) -> str:
    if expected:
        return "tp" if predicted else "fn"
    return "fp" if predicted else "tn"


def evaluate_release_intake(
    gold_path: Path,
    *,
    min_quality: float = CurationConfig().min_quality,
    quality_model: FastTextQuality | None = None,
    quality_calibration: dict[str, float] | None = None,
    max_rows: int = _MAX_GOLD_ROWS,
) -> dict:
    """Assess content gates independently using English, Tamil, or Hindi gold."""
    if not _threshold(min_quality):
        raise ValueError("min_quality must be within [0, 1]")
    if type(max_rows) is not int or not 1 <= max_rows <= _MAX_GOLD_ROWS:
        raise ValueError(f"max_rows must be 1..{_MAX_GOLD_ROWS}")
    if quality_model is None:
        if quality_calibration is not None:
            raise ValueError("quality calibration requires a quality model")
        thresholds = dict.fromkeys(_LANGUAGES, min_quality)
        model_metadata = None
    else:
        if (
            not isinstance(quality_calibration, dict)
            or quality_calibration.keys() != set(_LANGUAGES)
            or not all(_threshold(value) for value in quality_calibration.values())
        ):
            raise ValueError("quality calibration requires en, ta and hi thresholds within [0, 1]")
        thresholds = dict(quality_calibration)
        metadata = quality_model.metadata()
        model_metadata = {key: metadata[key] for key in (
            "family", "model_sha256", "license_declared"
        )}

    gold_path = Path(gold_path)
    if not gold_path.is_file():
        raise ValueError("gold samples must be a regular JSONL file")
    byte_count = 0
    rows = 0
    gold_hash = sha256()
    counts: dict[str, Counter] = {"privacy": Counter(), "quality": Counter()}
    by_language: dict[str, dict[str, Counter]] = {
        kind: defaultdict(Counter) for kind in counts
    }
    with gold_path.open("rb") as stream:
        for number, raw in enumerate(
            iter(lambda: stream.readline(_MAX_ROW_BYTES + 1), b""), 1
        ):
            byte_count += len(raw)
            if number > max_rows or byte_count > _MAX_GOLD_BYTES:
                raise ValueError("gold sample size cap exceeded")
            if len(raw) > _MAX_ROW_BYTES:
                raise ValueError(f"gold sample row {number} exceeds the byte limit")
            gold_hash.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (ValueError, RecursionError):
                raise ValueError(f"invalid gold sample JSON on line {number}") from None
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("text"), str)
                or row.get("language") not in _LANGUAGES
                or not ({"contains_pii", "should_keep"} & row.keys())
                or any(
                    label in row and type(row[label]) is not bool
                    for label in ("contains_pii", "should_keep")
                )
            ):
                raise ValueError(f"invalid labeled gold sample on line {number}")
            text = canonical_text(row["text"])
            if not text:
                raise ValueError(f"empty gold sample text on line {number}")
            language = row["language"]
            rows += 1
            if "contains_pii" in row:
                result = _outcome(row["contains_pii"], release_has_pii(text))
                counts["privacy"][result] += 1
                by_language["privacy"][language][result] += 1
            if "should_keep" in row:
                if quality_model is None:
                    score = score_text(text)[0]
                else:
                    try:
                        score = quality_model.keep_probability(text)
                    except (ValueError, RuntimeError, TypeError):
                        raise ValueError(f"quality model failed on gold line {number}") from None
                if not _threshold(score):
                    raise ValueError(f"quality model score is invalid on gold line {number}")
                result = _outcome(row["should_keep"], score >= thresholds[language])
                counts["quality"][result] += 1
                by_language["quality"][language][result] += 1
    if not rows:
        raise ValueError("gold sample file contains no labeled records")

    return {
        "schema_version": 1,
        "kind": "ada-release-independent-content-evaluation",
        "curator_version": VERSION,
        "gold_file_sha256": gold_hash.hexdigest(),
        "rows": rows,
        "quality_model": model_metadata,
        "quality_thresholds": thresholds,
        **{
            kind: {
                "labeled_rows": sum(tally.values()),
                "metrics": _confusion_metrics(tally),
                "metrics_by_language": {
                    language: _confusion_metrics(language_tally)
                    for language, language_tally in sorted(by_language[kind].items())
                },
            }
            for kind, tally in counts.items()
        },
        "interpretation": (
            "Privacy and quality are measured independently on their labeled samples. "
            "Use independently reviewed held-out annotations and read metrics in light "
            "of the sample composition."
        ),
    }


def add_assessment_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "assess-release", help="measure release privacy and quality using private human labels"
    )
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-quality", type=float, default=CurationConfig().min_quality)
    parser.add_argument("--quality-model", type=Path)
    parser.add_argument("--quality-model-sha256")
    parser.add_argument("--quality-model-license")
    parser.add_argument(
        "--quality-calibration", type=Path, help="JSON object of en/ta/hi keep thresholds"
    )


def assess_release_cli(args: argparse.Namespace) -> dict:
    model_flags = (
        args.quality_model,
        args.quality_model_sha256,
        args.quality_model_license,
        args.quality_calibration,
    )
    model = None
    calibration = None
    if any(value is not None for value in model_flags):
        if any(value is None for value in model_flags):
            raise ValueError("quality model, SHA-256, license, and calibration must be supplied together")
        if not args.quality_calibration.is_file():
            raise ValueError("quality calibration must be a regular JSON file")
        with args.quality_calibration.open("rb") as stream:
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError("quality calibration exceeds the byte limit")
        try:
            calibration = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            raise ValueError("quality calibration must be a UTF-8 JSON object") from None
        model = FastTextQuality(
            args.quality_model,
            expected_sha256=args.quality_model_sha256,
            declared_license=args.quality_model_license,
        )
    report = evaluate_release_intake(
        args.gold,
        min_quality=args.min_quality,
        quality_model=model,
        quality_calibration=calibration,
    )
    write_report(report, args.report)
    return report
