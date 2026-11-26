"""Fit and calibrate a local fastText binary quality model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from itertools import groupby
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable

from .core import canonical_text, file_digest, fingerprint, iter_hashed_lines, strict_json
from .evaluation import _confusion_metrics
from .release_intake import FastTextQuality, quality_model_input, release_has_pii

_LANGUAGES = ("en", "ta", "hi")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_FIT_PARAMETERS = {
    "lr": 0.5, "dim": 32, "epoch": 20, "wordNgrams": 2, "bucket": 50_000,
    "minCount": 1, "loss": "softmax", "seed": 2026, "thread": 1,
}


def _records(
    path: Path, expected: str, partition: str, max_rows: int,
    seen_text: set[str], fit_families: set[str],
    visit: Callable[[str, str, bool], None],
) -> dict:
    if path.stat().st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"{partition} labels exceed the byte limit")
    tally = {language: Counter(keep=0, drop=0) for language in _LANGUAGES}
    hasher = sha256()
    with path.open("rb") as stream:
        for number, (raw, _) in enumerate(
            iter_hashed_lines(stream, 512 * 1024, input_hasher=hasher), 1
        ):
            if number > max_rows:
                raise ValueError(f"{partition} labels exceed the row limit")
            try:
                record = strict_json(raw.decode("utf-8")) if raw is not None else None
            except (ValueError, RecursionError) as exc:
                raise ValueError(f"invalid {partition} label on line {number}") from exc
            if (
                not isinstance(record, dict)
                or not {"text", "language", "should_keep"} <= set(record)
                or set(record) - {"text", "language", "should_keep", "family_id"}
                or not isinstance(record["language"], str) or record["language"] not in tally
                or type(record["should_keep"]) is not bool
                or not isinstance(record["text"], str) or len(record["text"]) > 50_000
            ):
                raise ValueError(f"invalid {partition} label on line {number}")
            text = canonical_text(record["text"])
            model_text = quality_model_input(text)
            if not model_text or release_has_pii(text):
                raise ValueError(f"{partition} text must be nonempty and privacy-reviewed on line {number}")
            family = record.get("family_id")
            if "family_id" in record and (
                not isinstance(family, str) or not _FAMILY.fullmatch(family) or release_has_pii(family)
            ):
                raise ValueError(f"invalid {partition} family on line {number}")
            text_digest = fingerprint(model_text)
            if text_digest in seen_text or (partition == "calibration" and family in fit_families):
                raise ValueError(f"{partition} label repeats a normalized record or fit family")
            seen_text.add(text_digest)
            if partition == "fit" and family is not None:
                fit_families.add(family)
            language, label = record["language"], record["should_keep"]
            tally[language]["keep" if label else "drop"] += 1
            visit(text, language, label)
    if hasher.hexdigest() != expected:
        raise ValueError(f"{partition} labels SHA-256 mismatch")
    return {lang: dict(counts) for lang, counts in tally.items()}


def _calibrate(predictions: list[tuple[float, bool]]) -> tuple[float, dict]:
    """Choose maximum F1, then precision, then the highest threshold."""
    ordered = sorted(predictions, reverse=True)
    positives = sum(label for _, label in ordered)
    negatives = len(ordered) - positives
    best_rank = (-1.0, -1.0, -1.0)
    best_counts = Counter()
    tp = fp = 0
    for threshold, group in groupby(ordered, key=lambda entry: entry[0]):
        for _, label in group:
            if label:
                tp += 1
            else:
                fp += 1
        fn = positives - tp
        precision = tp / (tp + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        rank = (f1, precision, threshold)
        if rank > best_rank:
            best_rank = rank
            best_counts = Counter(tp=tp, fp=fp, fn=fn, tn=negatives - fp)
    return best_rank[2], _confusion_metrics(best_counts)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def train_quality_model(
    fit: Path, calibration: Path, output: Path, *, fit_sha256: str,
    calibration_sha256: str, model_license: str, min_calibration_per_class: int = 5,
    max_rows: int = 100_000,
) -> dict:
    """Publish a pinned binary model and aggregate calibration evidence."""
    if (
        not isinstance(fit_sha256, str) or not _SHA256.fullmatch(fit_sha256)
        or not isinstance(calibration_sha256, str) or not _SHA256.fullmatch(calibration_sha256)
        or fit_sha256 == calibration_sha256
    ):
        raise ValueError("fit and calibration require distinct pinned SHA-256 digests")
    if (
        type(max_rows) is not int or not 6 <= max_rows <= 100_000
        or type(min_calibration_per_class) is not int
        or not 1 <= min_calibration_per_class <= max_rows // 6
    ):
        raise ValueError("invalid quality training sample bounds")
    if (
        not isinstance(model_license, str) or not model_license.strip()
        or len(model_license) > 200 or release_has_pii(model_license)
    ):
        raise ValueError("quality training needs an operator-reviewed model license")
    if os.path.lexists(output):
        raise FileExistsError("quality training output already exists")
    try:
        import fasttext  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("install quality-model extras with pip install 'ada-datasets[models]'") from exc

    from .release import _publish

    output.parent.mkdir(parents=True, exist_ok=True)
    seen_text: set[str] = set()
    fit_families: set[str] = set()
    with tempfile.TemporaryDirectory(prefix=".ada-quality-fit-", dir=output.parent) as temporary:
        work = Path(temporary)
        stage = work / "output"
        stage.mkdir(mode=0o700)
        private_fit = work / "supervised.txt"
        with private_fit.open("w", encoding="utf-8", newline="\n") as stream:
            def write_fit(text: str, _language: str, should_keep: bool) -> None:
                label = "keep" if should_keep else "drop"
                stream.write(f"__label__{label} {quality_model_input(text)}\n")

            fit_counts = _records(
                fit, fit_sha256, "fit", max_rows, seen_text, fit_families, write_fit
            )
        if any(not counts["keep"] or not counts["drop"] for counts in fit_counts.values()):
            raise ValueError("fit requires both labels for each supported language")

        model_file = stage / "quality.bin"
        fasttext.train_supervised(input=str(private_fit), verbose=0, **_FIT_PARAMETERS).save_model(
            str(model_file)
        )
        model_sha = file_digest(model_file)
        scorer = FastTextQuality(model_file, expected_sha256=model_sha, declared_license=model_license)
        predictions = defaultdict(list)

        def score(text: str, language: str, should_keep: bool) -> None:
            predictions[language].append((scorer.keep_probability(text), should_keep))

        calibration_counts = _records(
            calibration, calibration_sha256, "calibration", max_rows, seen_text, fit_families, score
        )
        if any(
            min(counts.values()) < min_calibration_per_class for counts in calibration_counts.values()
        ):
            raise ValueError("calibration requires the minimum count for both labels per language")
        if file_digest(fit) != fit_sha256 or file_digest(calibration) != calibration_sha256:
            raise ValueError("labels changed while training the quality model")

        calibrated = {language: _calibrate(predictions[language]) for language in _LANGUAGES}
        fragment = {
            "path": "quality.bin", "sha256": model_sha, "license": model_license,
            "calibration": {language: result[0] for language, result in calibrated.items()},
        }
        try:
            library_version = version("fasttext-wheel")
        except PackageNotFoundError:
            library_version = version("fasttext")
        report = {
            "schema_version": 1,
            "kind": "ada-quality-model-calibration",
            "fasttext_version": library_version,
            "model": fragment,
            "parameters": _FIT_PARAMETERS,
            "fit": {"sha256": fit_sha256, "counts_by_language": fit_counts},
            "calibration": {
                "sha256": calibration_sha256,
                "minimum_per_label_per_language": min_calibration_per_class,
                "counts_by_language": calibration_counts,
                "metrics_by_language": {language: result[1] for language, result in calibrated.items()},
                "threshold_objective": "maximum F1, then precision, then highest threshold",
            },
            "evaluation": "Calibration selects thresholds. Use separate independent gold for final evaluation.",
        }
        _write_json(stage / "quality-model.json", fragment)
        _write_json(stage / "training-report.json", report)
        _publish(stage, output)
    return report


def add_quality_training_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train-quality", help="fit and calibrate a pinned local quality model")
    parser.add_argument("--fit", required=True, type=Path, help="private labeled training JSONL")
    parser.add_argument("--fit-sha256", required=True, help="SHA-256 of exact fit bytes")
    parser.add_argument("--calibration", required=True, type=Path, help="disjoint private calibration JSONL")
    parser.add_argument("--calibration-sha256", required=True, help="SHA-256 of exact calibration bytes")
    parser.add_argument("--model-license", required=True, help="operator-reviewed resulting model terms")
    parser.add_argument("--output", required=True, type=Path, help="new directory for model and reports")
    parser.add_argument("--min-calibration-per-class", type=int, default=5, help="per language")
    parser.add_argument("--max-rows", type=int, default=100_000, help="maximum rows per partition")
    return parser


def run_quality_training(args: argparse.Namespace) -> dict:
    return train_quality_model(
        args.fit, args.calibration, args.output,
        fit_sha256=args.fit_sha256, calibration_sha256=args.calibration_sha256,
        model_license=args.model_license, min_calibration_per_class=args.min_calibration_per_class,
        max_rows=args.max_rows,
    )
