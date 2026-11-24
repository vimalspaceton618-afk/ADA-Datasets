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


## v0.2: reviewable source rights, revocation lineage and measured diagnostics

**Review-date-pinned rights policy:** V2 source policies add
`review.reviewed_by`, `review.reviewed_at`, `review.evidence_sha256`,
`allowed_uses` (including `pretraining`), `commercial_allowed`,
`redistribution_allowed`, and `retention_until` (an ISO date or null).
These fields are declarations by your own reviewer, **not** automatic
legal clearance. For commercial model training add `--commercial-use`: legacy
v1 approvals are rejected, and v2 sources with `commercial_allowed: false`
are excluded. This flag cannot establish rights by itself. Policies remain local and may contain restricted details.

    ada-curate run --input examples/input.jsonl \
      --sources private/reviewed_v2_sources.json \
      --as-of-date 2026-11-25 --script-sanity \
      --output .runs/reviewed-2026-11-25

Version 1 source policy files remain readable for backward compatibility
but do not have evidence/retention enforcement. To use the stricter mode,
migrate each approved source to v2 and keep review records privately.
Script sanity checks Unicode writing systems for English, Tamil and Hindi;
it **cannot** identify the language and can reject legitimate code-mixing.

**Append-only successor snapshots for removals:** To remove a document
from a previously curated snapshot, create a private JSONL tombstone registry
with either `{"doc_id":"source:source-local-id"}` or
`{"sha256":"<lowercase SHA-256 of normalized text>"}`. Then:

    ada-curate revoke --parent .runs/reviewed-2026-11-25 \
      --tombstones private/tombstones.jsonl \
      --output .runs/reviewed-2026-11-25-revoked
    ada-curate verify --output .runs/reviewed-2026-11-25-revoked

A new snapshot is created, with matching documents removed, a new
`manifest.json`, the parent manifest's hash, and content-free removal
receipts. Existing snapshots are never overwritten. **Removal does not
erase the parent**, other dataset copies, model weights, GPU caches or
previously trained systems. Operators must explicitly quarantine/delete
all ancestor and downstream copies. Maintain the same tombstone registry
when subsequently rebuilding datasets from raw inputs.

**Corpus quality and independent human ratings:**

    ada-curate assess --snapshot .runs/reviewed-2026-11-25-revoked \
      --report .runs/quality-assessment.json

Optional `--labels private/independent_ratings.jsonl --threshold 0.75`
reads records with `sha256` and binary `quality_label` (0 or 1).
The JSON report contains aggregate score distributions and a confusion
matrix/precision/recall on *labeled accepted documents only*. It cannot
estimate rejection quality or incoming-pool quality without independent
sampling that includes rejected items. This is not a trained classifier.

The v0.2 implementation remains a **single-host research prototype**.
Review actual source terms, collect real annotations, validate multilingual
privacy performance, establish downstream deletion procedures, and run
controlled ADA pretraining ablations before relying on it at scale.

## v0.3: language-ID gates and independent evaluation

The existing --script-sanity option is not a language identifier. For model-based
language filtering, ADA Curator now supports a **local operator-supplied
original fastText lid.176** file (.bin or .ftz). The optional Python dependency
is installed separately with: python -m pip install fasttext

**Important license distinction:** The original fastText lid.176 models are
released under CC-BY-SA-3.0:
https://fasttext.cc/docs/en/language-identification.html

The similarly named Hugging Face facebook/fasttext-language-identification
repository hosts a **different NLLB model** and specifies CC-BY-NC-4.0:
https://huggingface.co/facebook/fasttext-language-identification

We neither download nor distribute model weights. A model must be supplied
locally along with its *actual SHA-256 digest* and declared license, subject
to independent rights review. Pinning a file cannot prove its license.

### Local model-based data curation

Obtain the original licensed weights yourself. Verify the file digest using
sha256sum on Linux/macOS, or Get-FileHash on Windows. Use your verified actual
SHA-256 in place of the placeholder:

    ada-curate run --input data/raw/approved.jsonl \
      --sources private/approved_v2_sources.json \
      --as-of-date 2026-11-25 --commercial-use \
      --lid-model private/models/lid.176.ftz \
      --lid-model-sha256 YOUR_REAL_64_CHARACTER_SHA256 \
      --lid-model-license CC-BY-SA-3.0 \
      --lid-min-confidence 0.85 \
      --output .runs/lid-gated-v03

A different model label or low-confidence result rejects the document and
records only a content-free reason. The manifest includes the pinned model
SHA-256 and its declared license. Scores are not calibrated error probabilities.

### Independent held-out language evaluation

A private JSONL gold file contains text and a human-verified language (en,
ta, hi or another ISO code, grouped as other). Run:

    ada-curate evaluate-lid --gold private/language_gold.jsonl \
      --model private/models/lid.176.ftz \
      --model-sha256 YOUR_REAL_64_CHARACTER_SHA256 \
      --model-license CC-BY-SA-3.0 \
      --min-confidence 0.85 \
      --report .runs/lid-evaluation.json

This reports aggregate confusion, abstentions and coverage. The repository
unit tests use **mock inference only**; no live real-model accuracy claim is made.

### Evaluate the full original curator input (including rejects)

The v0.2 assess command measures quality only for accepted documents.
Version 0.3 adds a full-pool evaluation. Supply the original byte-exact
input file and independent human labels indexed by 1-based line number,
the SHA-256 of the exact original JSONL line including newline, a Boolean
should_keep and optional independent language label.

    ada-curate evaluate-filter --snapshot .runs/original-curation \
      --input data/raw/approved.jsonl \
      --gold private/input_filter_gold.jsonl \
      --report .runs/curation-evaluation.json

The evaluator verifies the source checksum against the snapshot and
cross-checks every rejected audit-line hash. It reports aggregate confusion,
per-language metrics if labeled, precision, recall and approximate Wilson
95% intervals without logging any raw text. It refuses derived revocation
snapshots and does not infer performance on an unlabeled population.
A representative, independently annotated sample is required for meaningful
results. For full methodology see docs/LANGUAGE_AND_DATA_EVALUATION.md.
