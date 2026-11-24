# ADA Curator v0.1 — research, governance and limitations

## Public reference baselines (not private OpenAI/Anthropic procedures)

- FineWeb/FineWeb-Edu: https://arxiv.org/abs/2406.17557
- FineWeb2 multilingual considerations: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- DataComp-LM: https://arxiv.org/abs/2406.11794
- Allen AI Dolma 3 and toolkit: https://docs.allenai.org/training_data/dolma/index.html
- NVIDIA NeMo Curator 26.09: https://docs.nvidia.com/nemo/curator/v26.09/about/concepts/text/data/data-curation-pipeline
- Training dataset licensing and provenance audit: https://www.nature.com/articles/s42256-024-00878-8

## Original ADA Provenance Lattice (APL-v0.1)

Source rights approval precedes all content transformation. An input
record is eligible only when its source_id, license and reviewed rights_basis
match the operator-controlled registry. Opt-outs and takedown fingerprints
override source approvals.

Lattice means a sequence of independently auditable admissibility predicates
instead of a single opaque rating. The first sequence comprises provenance,
input integrity, explicit opt-outs and PII/secret exclusion, hygiene, declared
language, heuristic quality, benchmark decontamination, exact/fuzzy dedup and
stable family-level train/validation/test assignment. Rejections enter a
content-free audit JSONL with source_id, input-line hash and reason.

A new design and implementation does not imply patentability or demonstrate
superiority to private research labs. This repository is public and Apache-2.0;
any confidential weighting recipes, source contracts, key material, private
datasets and secret models must remain in protected infrastructure.

## Source rights

Public accessibility, the label on a dataset card, or someone's earlier model
training experiment does not necessarily establish commercial training or
redistribution rights. Review the underlying corpus and applicable law.
The example policy approves only newly written demonstration text. Ingesting
a Hub dataset requires an independently approved policy entry and exact
revision pinning. The HF adapter requires an explicit rights acknowledgement.

Maintain a private dataset registry with contract IDs, retention dates, named
reviewers, takedown contact and licensing obligations. Version 0.1 does not
solve legal compliance automatically.

## Engineering limits requiring later milestones

- Near duplicates use deterministic 32-hash MinHash LSH (8 bands of 4),
  then actual 5-token shingle Jaccard for candidates. LSH has false negatives;
  this is not exhaustive deduplication or semantic near-duplicate removal.
- en/ta/hi are declared by the input provider, not verified language ID.
  Scores are not calibrated across languages, especially Tamil/Hindi.
- Heuristic quality is not educational value. No learned, independently
  assessed quality models have been trained or evaluated.
- Emails, common phone numbers and obvious private-key patterns are rejected,
  but heuristic PII filtering can miss sensitive data or falsely reject text.
- Benchmark decontamination is exact SHA-256 or bounded 5-gram Jaccard;
  paraphrases, hidden benchmarks and longer-context leakage may survive.
- No automatic HTML extraction, PDF parsing, code licensing, robots controls,
  crawl policies, malware scanning or source-to-document legal verification.
- Same ordered input bytes plus config give identical output bytes, but changing
  input order may change which near-duplicate representative is retained.
- SQLite and Python loops are single-host reference components. A production
  deployment needs sharded distributed dedup, storage transactions, cluster
  recovery, learned scoring and calibrated multilingual validation.
- Raw input may contain PII. Work on trusted local drives; do not commit raw
  samples or private policy registries to public GitHub.
- Quality reports do not demonstrate ADA model performance. Controlled
  pretraining ablations against fixed baselines are required.

## Release gates before expensive ADA training

1. Verify actual source rights and attribution/retention requirements.
2. Establish a documented withdrawal and dataset removal process.
3. Generate and independently review a content-free provenance manifest.
4. Audit PII, safety, benchmark leakage and source/dataset overlap.
5. Evaluate coverage and quality across languages/domains with human labels.
6. Run fixed-budget model training ablations and statistical confidence checks.
7. Document data composition, versions and model-training evidence.
