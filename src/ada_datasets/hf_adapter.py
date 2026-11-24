"""Optional bounded Hugging Face streaming adapter.

This module never downloads on import and does not equate Hub dataset labels
with legal clearance of upstream material. Output is RAW, uncurated and private
until run through ADA Curator.
"""
from __future__ import annotations

from hashlib import sha256
import os
import re
import tempfile

from .core import load_policy, write_jsonl


def adapt_hf(args) -> dict:
    if not args.acknowledge_source_rights:
        raise ValueError("must --acknowledge-source-rights before fetching")
    if not (1 <= args.max_rows <= 100_000):
        raise ValueError("max-rows must be 1..100000 in v0.1")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("pin an exact 40-hex-character Hugging Face revision")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing raw export")
    policies = load_policy(args.sources)
    policy = policies.get(args.source_id)
    if policy is None or not policy["training_allowed"]:
        raise ValueError("source is not approved in policy")
    for key, expected in (
        ("hf_dataset_id", args.dataset_id), ("hf_config", args.config),
        ("hf_split", args.split), ("hf_revision", args.revision),
    ):
        if policy.get(key) != expected:
            raise ValueError(f"policy lacks exact HF source binding: {key}")
    if args.language not in ("en", "ta", "hi"):
        raise ValueError("unsupported declared language")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("install optional dependency: pip install datasets") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".ada-hf-", suffix=".jsonl",
                                    dir=args.output.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            ds = load_dataset(args.dataset_id, name=args.config, split=args.split,
                              revision=args.revision, streaming=True)
            for record in ds:
                if count == args.max_rows:
                    break
                if not isinstance(record.get("text"), str):
                    continue
                # Hash upstream identifiers so signed URLs and personal
                # query parameters do not enter generated document IDs.
                original_id = (record.get("id") or record.get("url") or
                               sha256(record["text"].encode()).hexdigest())
                document_id = sha256(
                    (args.dataset_id + ":" + str(original_id)).encode()
                ).hexdigest()
                language = record.get("language") or args.language
                if language not in ("en", "ta", "hi"):
                    continue
                write_jsonl(stream, {
                    "id": document_id, "text": record["text"],
                    "source_id": args.source_id,
                    "license": policy["license"],
                    "rights_basis": policy["rights_basis"],
                    "language": language, "url": record.get("url"),
                    "family_id": document_id,
                })
                count += 1
        os.rename(temporary, args.output)
        return {"exported_rows": count, "path": str(args.output),
                "status": "RAW_UNCURATED_REQUIRES_ADA_CURATE_RUN"}
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
