"""Command-line interface for ADA's source-registered data curator."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .core import CurationConfig, curate_dataset, file_digest


def validate_output(directory: Path, expected_manifest_sha256: str | None = None) -> dict:
    manifest_file = directory / "manifest.json"
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise ValueError("missing regular manifest")
    raw = manifest_file.read_bytes()
    manifest_sha256 = sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("manifest digest mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema_version") == 2:
        from .release import verify_release
        return verify_release(directory, manifest_sha256)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    expected = {"train.jsonl", "validation.jsonl", "test.jsonl", "audit.jsonl"}
    if set(manifest.get("outputs_sha256", {})) != expected:
        raise ValueError("manifest is missing required output hashes")
    splits = {}
    families: dict[str, str] = {}
    for name in sorted(expected):
        file = directory / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"missing or symlinked output: {name}")
        if file_digest(file) != manifest["outputs_sha256"][name]:
            raise ValueError(f"corrupt output: {name}")
        count = 0
        with file.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if name == "audit.jsonl":
                    if "text" in row or row.get("decision") != "rejected":
                        raise ValueError("audit must never contain raw training text")
                else:
                    if row.get("split") != name.removesuffix(".jsonl"):
                        raise ValueError("split label mismatch")
                    if sha256(row["text"].encode("utf-8")).hexdigest() != row["sha256"]:
                        raise ValueError("document text digest mismatch")
                    unique = row["sha256"]
                    if unique in splits:
                        raise ValueError("cross-split exact duplication")
                    splits[unique] = name
                    family = row["source_id"] + ":" + row["family_id"]
                    if family in families and families[family] != name:
                        raise ValueError("family group crosses train/validation/test")
                    families[family] = name
                count += 1
        if name != "audit.jsonl" and count != manifest["accepted_splits"][name.removesuffix(".jsonl")]:
            raise ValueError("split row count differs from manifest")
    if len(splits) != manifest["totals"].get("accepted", 0):
        raise ValueError("accepted row count differs from manifest")
    if "revocation_receipt_sha256" in manifest:
        receipt = directory / "revoked.jsonl"
        if receipt.is_symlink() or not receipt.is_file():
            raise ValueError("missing or symlinked revocation receipt")
        if file_digest(receipt) != manifest["revocation_receipt_sha256"]:
            raise ValueError("corrupt revocation receipt")
        count = 0
        with receipt.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (not isinstance(row, dict) or
                        set(row) != {"decision", "split", "row_sha256"} or
                        row["decision"] != "removed" or
                        row["split"] not in ("train", "validation", "test") or
                        not isinstance(row["row_sha256"], str) or
                        len(row["row_sha256"]) != 64 or
                        any(c not in "0123456789abcdef" for c in row["row_sha256"])):
                    raise ValueError("invalid revocation receipt record")
                count += 1
        lineage = manifest.get("lineage", {})
        if count != lineage.get("removed_total", -1):
            raise ValueError("revocation receipt count differs from lineage")
        for name in ("parent_manifest_sha256", "tombstone_sha256"):
            digest = lineage.get(name)
            if not isinstance(digest, str) or len(digest) != 64 or any(
                c not in "0123456789abcdef" for c in digest
            ):
                raise ValueError(f"invalid revocation lineage {name}")
    return {
        "valid": True,
        "docs": len(splits),
        "manifest_sha256": manifest_sha256,
        "curator_version": manifest["curator_version"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ada-curate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    curate = sub.add_parser("run", help="curate source-registered local JSONL")
    curate.add_argument("--input", type=Path, required=True)
    curate.add_argument("--sources", type=Path, required=True,
                        help="operator-reviewed source rights policy JSON")
    curate.add_argument("--output", type=Path, required=True)
    curate.add_argument("--holdout", type=Path, help="benchmark holdout JSONL")
    curate.add_argument("--denylist", type=Path, help="doc_id or sha256 JSONL takedown registry")
    curate.add_argument("--seed", type=int, default=2026)
    curate.add_argument("--min-quality", type=float, default=0.65)
    curate.add_argument("--min-chars", type=int, default=100)
    curate.add_argument("--min-words", type=int, default=18)
    curate.add_argument("--near-jaccard", type=float, default=0.85)
    curate.add_argument("--as-of-date", help="YYYY-MM-DD: required for v2 rights expiry; pinned in manifest")
    curate.add_argument("--script-sanity", action="store_true",
                        help="conservative Unicode script sanity (not language identification)")
    curate.add_argument("--commercial-use", action="store_true",
                        help="require explicitly approved commercial use in v2 source registry")
    curate.add_argument("--lid-model", type=Path,
                        help="local original fastText lid.176.bin/.ftz; never downloaded automatically")
    curate.add_argument("--lid-model-sha256",
                        help="expected full SHA256 of exact LID model file")
    curate.add_argument("--lid-model-license",
                        help="operator-reviewed license of LID model: CC-BY-SA-3.0")
    curate.add_argument("--lid-min-confidence", default=0.80, type=float,
                        help="reject LID scores below this threshold")

    revoke = sub.add_parser("revoke", help="publish a new dataset snapshot excluding tombstones")
    revoke.add_argument("--parent", required=True, type=Path,
                        help="already curated and verified original snapshot")
    revoke.add_argument("--tombstones", required=True, type=Path,
                        help="private JSONL document identifiers or normalized SHA256 hashes")
    revoke.add_argument("--output", required=True, type=Path,
                        help="NEW output snapshot directory; never overwrite parent")

    assess = sub.add_parser("assess", help="aggregate corpus quality and optional human-label metrics")
    assess.add_argument("--snapshot", required=True, type=Path)
    assess.add_argument("--labels", type=Path, help="independent SHA-256/quality_label JSONL")
    assess.add_argument("--threshold", default=0.75, type=float)
    assess.add_argument("--report", required=True, type=Path, help="new JSON report output")

    eval_lid = sub.add_parser("evaluate-lid",
        help="evaluate optional local language ID on independent gold-labeled examples")
    eval_lid.add_argument("--gold", type=Path, required=True)
    eval_lid.add_argument("--model", type=Path, required=True)
    eval_lid.add_argument("--model-sha256", required=True)
    eval_lid.add_argument("--model-license", required=True)
    eval_lid.add_argument("--min-confidence", type=float, default=0.80)
    eval_lid.add_argument("--report", type=Path, required=True)

    eval_filter = sub.add_parser("evaluate-filter",
        help="measure accepted/rejected decisions against independent line-level labels")
    eval_filter.add_argument("--snapshot", type=Path, required=True)
    eval_filter.add_argument("--input", type=Path, required=True,
                             help="exact original raw JSONL used by run")
    eval_filter.add_argument("--gold", type=Path, required=True)
    eval_filter.add_argument("--report", type=Path, required=True)

    verify = sub.add_parser("verify", help="verify immutable manifest/output integrity")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256")

    release = sub.add_parser("release", help="publish or resume a verified ADA training release")
    release.add_argument("--recipe", type=Path, required=True)
    release.add_argument("--work-dir", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--workers", type=int, default=1,
                         help="local workers; set to zero to use completed remote tasks")
    release.add_argument("--prepare-only", action="store_true",
                         help="initialize pinned tasks for external workers")
    release.add_argument("--parent", type=Path, help="verified release to revoke and supersede")
    release.add_argument("--status", type=Path, help="private initialized governance registry")

    worker = sub.add_parser("worker", help="run one resumable curation task")
    worker.add_argument("--work-dir", type=Path, required=True)
    worker.add_argument("--task-id", type=int, required=True)

    from .release_assessment import add_assessment_parser
    from .release_governance import add_governance_commands
    from .quality_training import add_quality_training_parser
    add_assessment_parser(sub)
    add_governance_commands(sub)
    add_quality_training_parser(sub)

    hf = sub.add_parser("adapt-hf", help="approved and bounded Hugging Face streaming adapter")
    hf.add_argument("--sources", type=Path, required=True)
    hf.add_argument("--source-id", required=True)
    hf.add_argument("--dataset-id", required=True)
    hf.add_argument("--config", required=True)
    hf.add_argument("--split", required=True)
    hf.add_argument("--revision", required=True, help="pinned full 40-character git SHA")
    hf.add_argument("--language", required=True, help="operator-declared, not language identification")
    hf.add_argument("--max-rows", required=True, type=int)
    hf.add_argument("--output", required=True, type=Path)
    hf.add_argument("--acknowledge-source-rights", action="store_true",
                    help="confirm independent review of license and content rights")

    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = validate_output(args.output, args.expected_manifest_sha256)
        elif args.command == "release":
            from .release import run_release
            result = run_release(args.recipe, args.work_dir, args.output,
                                 workers=args.workers, prepare_only=args.prepare_only,
                                 parent=args.parent, status=args.status)
        elif args.command == "worker":
            from .release import run_worker
            result = run_worker(args.work_dir, args.task_id)
        elif args.command == "governance":
            from .release_governance import run_governance_command
            result = run_governance_command(args)
        elif args.command == "assess-release":
            from .release_assessment import assess_release_cli
            result = assess_release_cli(args)
        elif args.command == "train-quality":
            from .quality_training import run_quality_training
            result = run_quality_training(args)
        elif args.command == "revoke":
            from .lineage import revoke_snapshot
            result = revoke_snapshot(args.parent, args.tombstones, args.output)
        elif args.command == "assess":
            from .quality import evaluate_quality, write_report
            result = evaluate_quality(args.snapshot, labels_path=args.labels,
                                      threshold=args.threshold)
            write_report(result, args.report)
        elif args.command == "adapt-hf":
            from .hf_adapter import adapt_hf
            result = adapt_hf(args)
        elif args.command == "evaluate-lid":
            from .evaluation import evaluate_language_id
            from .lid import FastTextLID176
            from .quality import write_report
            identifier = FastTextLID176(
                args.model, expected_sha256=args.model_sha256,
                declared_license=args.model_license,
            )
            result = evaluate_language_id(
                identifier, args.gold, min_confidence=args.min_confidence
            )
            write_report(result, args.report)
        elif args.command == "evaluate-filter":
            from .evaluation import evaluate_filter
            from .quality import write_report
            result = evaluate_filter(args.snapshot, args.input, args.gold)
            write_report(result, args.report)
        else:
            lid_args = (args.lid_model, args.lid_model_sha256, args.lid_model_license)
            if any(x is not None for x in lid_args) and not all(
                x is not None for x in lid_args
            ):
                raise ValueError(
                    "--lid-model, --lid-model-sha256 and --lid-model-license "
                    "must be supplied together"
                )
            identifier = None
            if args.lid_model is not None:
                from .lid import FastTextLID176
                identifier = FastTextLID176(
                    args.lid_model, expected_sha256=args.lid_model_sha256,
                    declared_license=args.lid_model_license
                )
            config = CurationConfig(seed=args.seed, min_quality=args.min_quality,
                                    min_chars=args.min_chars, min_words=args.min_words,
                                    min_near_jaccard=args.near_jaccard,
                                    script_sanity=args.script_sanity,
                                    lid_min_confidence=args.lid_min_confidence,
                                    commercial_use=args.commercial_use,
                                    as_of_date=args.as_of_date)
            result = curate_dataset(args.input, args.sources, args.output,
                                    config=config, holdout_path=args.holdout,
                                    denylist_path=args.denylist,
                                    language_identifier=identifier)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ada-curate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
