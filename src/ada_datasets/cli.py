"""Command-line interface for ADA's source-registered data curator."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from .core import CurationConfig, curate_dataset, file_digest


def validate_output(directory: Path) -> dict:
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
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
        "manifest_sha256": file_digest(manifest_file),
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

    verify = sub.add_parser("verify", help="verify immutable manifest/output integrity")
    verify.add_argument("--output", type=Path, required=True)

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
            result = validate_output(args.output)
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
        else:
            config = CurationConfig(seed=args.seed, min_quality=args.min_quality,
                                    min_chars=args.min_chars, min_words=args.min_words,
                                    min_near_jaccard=args.near_jaccard,
                                    script_sanity=args.script_sanity,
                                    commercial_use=args.commercial_use,
                                    as_of_date=args.as_of_date)
            result = curate_dataset(args.input, args.sources, args.output,
                                    config=config, holdout_path=args.holdout,
                                    denylist_path=args.denylist)
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"ada-curate: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
