"""Non-leaking corpus quality diagnostics and optional human-label evaluation.

Aggregate distribution counts and score statistics are descriptive, not an
evidence-backed quality classifier. Labels must be collected independently
and may cover only the accepted subset, never an unbiased incoming pool.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import statistics

from .core import file_digest

_SPLITS = ("train", "validation", "test")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 5),
        "min": round(min(values), 5),
        "max": round(max(values), 5),
    }


def _read_labels(path: Path | None) -> tuple[dict[str, int], str | None]:
    if path is None:
        return {}, None
    labels = {}
    with Path(path).open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid label row {n}")
            fingerprint = record.get("sha256")
            value = record.get("quality_label")
            if (not isinstance(fingerprint, str) or
                    len(fingerprint) != 64 or
                    any(c not in "0123456789abcdef" for c in fingerprint) or
                    type(value) is not int or value not in (0, 1)):
                raise ValueError(f"invalid sha256/quality_label on line {n}")
            if fingerprint in labels:
                raise ValueError(f"duplicate human-label fingerprint on line {n}")
            labels[fingerprint] = value
    return labels, file_digest(Path(path))


def evaluate_quality(snapshot: Path, *, labels_path: Path | None = None,
                     threshold: float = 0.75) -> dict:
    """Evaluate score threshold vs independent labels *on accepted docs only*."""
    from .cli import validate_output
    snapshot = Path(snapshot)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be within [0,1]")
    validate_output(snapshot)
    labels, label_hash = _read_labels(labels_path)
    by_language: dict[str, list[float]] = defaultdict(list)
    by_split: dict[str, list[float]] = {name: [] for name in _SPLITS}
    by_domain = Counter()
    selected = set()
    confusion = Counter()
    for split in _SPLITS:
        with (snapshot / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                score = float(record["quality_score"])
                by_language[record["language_declared"]].append(score)
                by_split[split].append(score)
                by_domain[record.get("domain_declared", "unlabeled")] += 1
                key = record["sha256"]
                if key in labels:
                    selected.add(key)
                    predicted = score >= threshold
                    truth = bool(labels[key])
                    confusion[("tp" if predicted else "fn") if truth else (
                        "fp" if predicted else "tn")] += 1
    tp, fp, tn, fn = (confusion[k] for k in ("tp", "fp", "tn", "fn"))
    def fraction(n: int, denominator: int) -> float | None:
        return round(n / denominator, 6) if denominator else None
    return {
        "schema_version": 1,
        "kind": "ADA-Curator-accepted-corpus-quality-evaluation",
        "snapshot_manifest_sha256": file_digest(snapshot / "manifest.json"),
        "human_labels_sha256": label_hash,
        "quality_threshold": threshold,
        "scores_by_language": {key: _stats(vals) for key, vals in
                               sorted(by_language.items())},
        "scores_by_split": {key: _stats(vals) for key, vals in
                            sorted(by_split.items())},
        "domain_counts": dict(sorted(by_domain.items())),
        "human_labels": {
            "provided": len(labels), "matched": len(selected),
            "unmatched": len(labels) - len(selected),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": fraction(tp, tp + fp),
            "recall": fraction(tp, tp + fn),
            "f1": fraction(2 * tp, 2 * tp + fp + fn),
            "negative_recall": fraction(tn, tn + fp),
        },
        "limitations": [
            "Metrics cover only curated/accepted documents, so cannot estimate "
            "end-to-end filtering precision/recall on all incoming documents.",
            "Documents are not randomly sampled or human-rated unless an "
            "independent balanced annotation set is supplied.",
            "Scores come from simple heuristics and are not a trained classifier.",
        ],
    }


def write_report(report: dict, destination: Path) -> None:
    """Write a new report; never overwrite a previous evaluation."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
