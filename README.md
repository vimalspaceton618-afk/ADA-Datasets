# ADA-Datasets

**ADA Curator v0.1 — the ARcHi provenance-first training data workbench.**

This is an original, locally executable Python pipeline for curating **licensed
and operator-approved** language-model training records, built for the ADA model
research program. It is a working **single-host reference implementation**,
not a trillion-token production service, a copied private-lab pipeline, or
proof of proprietary technology.

## Original design: ADA Provenance Lattice (APL)

Source registry and consent/rights checks -> stable Unicode normalization ->
opt-out and secret/PII gate -> hygiene and heuristic quality -> benchmark
decontamination -> exact SHA-256 dedup -> MinHash-LSH + verified Jaccard fuzzy
dedup -> stable family-group train/validation/test split -> audit and
content-addressed manifest.

Only **approved** sources can pass. The output is local and immutable;
raw data is never uploaded to GitHub by the program.

## Installation

Requires Python 3.10+ with no mandatory third-party runtime dependencies.

    python -m pip install -e .

## Run the local smoke example

    ada-curate run --input examples/input.jsonl \
      --sources examples/sources.json \
      --output .runs/demo-v1

    ada-curate verify --output .runs/demo-v1

The output includes train.jsonl, validation.jsonl, test.jsonl, audit.jsonl and
manifest.json. No network/GPU/API key is needed for this demo. Re-run to a
**new output directory**; existing output directories are never overwritten.

## Input contract

Each input line is a JSON object containing:

- id: stable source-local document identifier
- source_id: must match an approved registry entry
- text: raw textual content, handled only on trusted machines
- license, rights_basis: must match that approved source entry exactly
- language: declared en, ta or hi (not independently verified)
- optional family_id: related pages/books/families share a split
- optional url, opt_out or do_not_train

Example records and source policy are under examples/. A source policy must
explicitly mark training_allowed true and have an operator-reviewed
rights_basis (owned, licensed, public_domain or open_dataset_approved).
This is a *policy gate*, not legal advice or automated verification.

Optional guards:

    ada-curate run --input input.jsonl --sources private/sources.json \
      --holdout private/benchmark.jsonl --denylist private/takedowns.jsonl \
      --output .runs/experiment-001 --min-quality 0.7

A benchmark file is JSONL with a text field. A denylist file is JSONL with
doc_id (source_id:id) or sha256 (digest of normalized text). Rejected records
are audited by line hash/reason without copying their raw text.

## Optional Hugging Face adapter

Only use this after **independent source-rights approval**. Create a source
entry in the registry binding source_id, license, rights_basis, training_allowed,
hf_dataset_id, hf_config, hf_split and hf_revision to **exact** verified values.
You must review the dataset's underlying terms, not only the Hub badge.
Install the optional dependency separately:

    python -m pip install datasets

Then run, substituting the policy-approved dataset, config and full commit SHA:

    ada-curate adapt-hf --sources private/approved_sources.json \
      --source-id my-reviewed-source \
      --dataset-id reviewed-organization/reviewed-dataset \
      --config sample --split train \
      --revision 0123456789abcdef0123456789abcdef01234567 \
      --language en --max-rows 1000 \
      --acknowledge-source-rights --output data/raw/reviewed.jsonl

The SHA above is an **example placeholder**. This exports *uncurated raw*
records locally. Afterwards run ada-curate run on them; do not commit to Git.

## Reproducibility and limitations

Manifest files record ordered input SHA-256, rights policy SHA-256, optional
holdout/denylist digests, config, per-stage reason counts and output digests.
The verify command checks output digests, content hashes and family split
exclusivity. Source metadata and rights must still be independently reviewed.

This v0.1 uses heuristic quality and declared language. PII regexes and
MinHash-LSH are fallible. Source rights are not automatically cleared, no
model training evidence is claimed, and the current CPU/SQLite pipeline is
not production-scale. Read docs/RESEARCH_AND_POLICY.md for all limitations.

## Roadmap

v0.2: source registries and audit DB, consent/removal propagation, language ID,
language-calibrated filters, domain labels, contamination reports.
v0.3: sharded/distributed dedup, multilingual/multimodal curation,
model-based quality filters and deterministic resume.
v0.4: human-rated controlled ablations, blending, tokenization and native ADA
training data contracts.

## Intellectual property

Copyright (c) ARcHi contributors. Source code is Apache-2.0, as defined by the
existing LICENSE file. Because this repository is public, code published here
is **not secret/proprietary**. Private heuristics or commercial policies should
be developed in a separate access-controlled project. Branding APL does not
imply an existing patent or an independently validated novel algorithm.
