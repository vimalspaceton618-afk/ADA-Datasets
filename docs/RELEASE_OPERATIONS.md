# Operate an ADA training release

ADA Curator v0.4 publishes immutable document and token shards on private Linux storage. The coordinator and workers need the same ADA Curator version, Parquet and tokenizer dependency versions, and absolute paths for the recipe and its assets. Use a volume that supports advisory `flock`, atomic directory rename without replacement, and fsync. Verify these properties across hosts before placing a real corpus on the volume. Give curation workers access to raw intake and work files. Give the trainer read access to the curated release and permission to record checkpoints in the governance registry.

## Build and verify the fixture

Run from an ADA-Datasets checkout with the matching release environment. Use a fresh private directory for each distinct recipe and its output.

```sh
python -m pip install '.[release]'
umask 077
pilot_root=$(mktemp -d)
python examples/make_release_fixture.py --output-dir "$pilot_root/input"
ada-curate governance --status "$pilot_root/status.json" init
ada-curate release --recipe "$pilot_root/input/recipe.json" \
  --work-dir "$pilot_root/work" --output "$pilot_root/release" \
  --workers 2 --status "$pilot_root/status.json"
release_sha=$(sha256sum "$pilot_root/release/manifest.json" | cut -d ' ' -f 1)
ada-curate verify --output "$pilot_root/release" --expected-manifest-sha256 "$release_sha"
```

For operational runs, record the verified manifest digest in the private handoff alongside the dependency versions and status-registry location. The trainer uses that independently recorded digest. The fixture command computes it from the directory for convenience. Review the dataset card, split coverage, expected token-budget shortfalls, rejection counts, model pins, and artifact inventory before handing off a real release.

In the companion ADA environment, install the matching ADA-Datasets wheel with its `release` extra and install ADA with its pretraining dependencies. From the ADA checkout, use the same release directory and registry values.

```sh
python scripts/ada_pretrain_fineweb_edu.py --device cpu --model-profile smoke \
  --curated-release "$pilot_root/release" --curated-release-sha256 "$release_sha" \
  --release-status-registry "$pilot_root/status.json" --output-dir "$pilot_root/checkpoints" \
  --max-steps 2 --batch-size 1 --save-every 1 --eval-every 1 --eval-batches 1
```

## Run work across hosts

1. Prepare a schema v1 recipe using the fixture layout. Give each input JSONL file a unique SHA-256 pin. Pin the schema v2 source policy, tokenizer, optional holdouts and denylist, and approved local model files. Set a review date in `curation.as_of_date`. Include the approval and attribution metadata, terms evidence, retention decision, and opt-out contact in the private registry. Set `release_use` to the actual intended use.
2. Run `ada-curate release --recipe RECIPE --work-dir WORK --output RELEASE --prepare-only` on the coordinator. Use the reported task count to dispatch `ada-curate worker --work-dir WORK --task-id N` for each zero-based task ID on the shared volume.
3. Run the same `release` command with `--workers 0 --status STATUS` in place of `--prepare-only` when all tasks finish. Workers verify input pins. The coordinator verifies task receipts and final artifact hashes before publication, and registers the published manifest with the initialized status registry.

Monitor worker exit status and receipt completion, input and output bytes, elapsed time, free bytes and inodes on both the work and release volumes, and coordinator memory. Worker processing is distributed by input file. The coordinator performs corpus-wide deduplication and packing, so size its disk and time separately. Keep task outputs, scratch space, source policy, hashes, and model evidence private.

Retry an interrupted worker with its original task ID and retry the coordinator with its original work directory and recipe. A valid completed task can be reused. If its receipts or payloads fail verification, retain the evidence securely and restart with a fresh work directory. A changed input or recipe requires a new work directory. For a failed cross-shard candidate limit, review corpus skew and duplicated boilerplate across the full input before building a new recipe. A failed join across shards cannot establish split isolation.

## Quarantine and replace a release

The status registry records release lineage, checkpoint digests, downstream copies, and deletion-receipt digests. Store it on the private shared volume, and save each reviewed registry head in a separate access-controlled audit system. Use `ada-curate governance --status STATUS verify --expected-head-sha256 HEAD` to match a known head. Hash chaining detects edits to existing records. An independently retained head detects replacement with a different history.

For each stored copy, create a private inventory record with a unique copy reference, location, and responsible operator. Register its digest using `ada-curate governance --status STATUS register-copy --copy-sha256 COPY --artifact-kind release --artifact-sha256 RELEASE_SHA` or `--artifact-kind checkpoint`. Include storage replicas, caches, and backups in this inventory.

ADA registers the SHA-256 of each immutable `checkpoint.json` index, which binds the checkpoint files. Resume repairs an interrupted registration for an active release. During quarantine inventory, reconcile any previously published checkpoint indexes that missed registration with `ada-curate governance --status STATUS register-checkpoint --checkpoint-sha256 CHECKPOINT_SHA --release-sha256 RELEASE_SHA`.

On receipt of a valid withdrawal, store the protected request and identify every release root in scope. Run `ada-curate governance --status STATUS quarantine --release-sha256 RELEASE_SHA --tombstone-sha256 REQUEST_SHA` to block registered releases and descendants. The command reports affected release, checkpoint, and copy digests. Stop the affected jobs and revoke access to the quarantined files in storage, including files held by readers that started before quarantine.

Add the withdrawal to the cumulative denylist and pin the new file in a successor recipe. Run `ada-curate release --recipe SUCCESSOR_RECIPE --work-dir SUCCESSOR_WORK --output SUCCESSOR --parent RELEASE --status STATUS`. The coordinator validates the parent, expands exact document aliases, and publishes a newly verified replacement. Use its `revocation-impact.json` to locate affected parent document and token shards. Hand off the replacement manifest digest for new training.

After a responsible operator verifies removal or records approved retention restrictions for an affected copy, preserve the evidence privately. For completed deletion, record the receipt with `ada-curate governance --status STATUS receipt --artifact-kind copy --artifact-sha256 COPY --receipt-sha256 RECEIPT`. Use the corresponding `release` or `checkpoint` kind for primary artifacts. `governance verify` reports pending deletions. Review checkpoint disposal with the model owner because physical deletion and model unlearning are distinct operations.
