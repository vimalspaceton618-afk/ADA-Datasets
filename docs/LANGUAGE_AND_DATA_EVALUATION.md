# ADA Curator v0.3 — research evaluation and licensing

## The two independent quality questions

1. Does the pinned language-ID model correctly identify English, Tamil, Hindi
   and out-of-scope languages on held-out, human-reviewed examples? Report
   confusion, coverage, abstentions and per-language recall, including mixed
   language/romanized text and Marathi/Nepali negative controls.
2. Does the entire curation pipeline retain high-quality examples and reject
   low-quality or unapproved examples? Require the original byte-exact input,
   original snapshot, and independent line-level human decisions. Report
   precision, recall, specificity and Wilson intervals for the labeled sample.

Earlier accepted-only reports cannot estimate rejection false negatives.
The v0.3 full-input evaluator addresses this by verifying original-line
checksums against the rejection log and independent human gold labels.
It does not certify the representativeness of the annotation sample.

## Licensing and source integrity

The original 176-language fastText weights are distributed by fastText
under CC-BY-SA-3.0; see:
https://fasttext.cc/docs/en/language-identification.html

The separate NLLB/Hugging Face fastText language-identification model is
marked CC-BY-NC-4.0:
https://huggingface.co/facebook/fasttext-language-identification

This repository does not redistribute pretrained model files. It requires an
explicit local original lid.176 model path, the operator's expected SHA-256,
and the CC-BY-SA-3.0 license declaration. A SHA digest alone cannot
establish commercial rights or confirm the true origin of an arbitrary file.
Legal/commercial rights review remains external and independently documented.

## Recommended independent annotation protocol

- Sample from all incoming sources and both accepted and rejected stages.
- Use two independent annotators per language and quality label, adjudicate
  disagreements and record inter-rater agreement in private systems.
- Partition the development, calibration and final evaluation sets by source;
  avoid near-duplicate and source-family overlap.
- Include Tamil script, Devanagari Hindi, code-mixed and transliterated
  examples, English-adjacent languages, very short excerpts and false friends.
- Fix thresholds on the development set and inspect final heldout metrics
  without repeatedly tuning to the benchmark.
- Document source rights, language balance, domain balance, and other
  selection biases before interpreting overall accuracy.
- Treat model scores as uncalibrated; threshold sweeps should include
  precision, recall, acceptance coverage, and abstention.

## Technical limitations

- Tests use deterministic fake language predictions; they validate logic
  but do not evaluate fastText accuracy on real Tamil/Hindi/English corpora.
- Language-ID scores are not guaranteed calibrated probabilities.
- Labels in benchmark JSONL must be independently created on permitted data.
- Accuracy and confidence bounds apply only to the sampled population, not
  to unobserved sources or examples with correlated duplicates.
- Full-input evaluation requires original raw data and an original snapshot;
  derived revocation snapshots need different accounting.
- Report files contain counts, SHA-256 digests and model metadata only, not
  individual inputs, identifiers or predictions.
- This remains a single-host SQLite reference pipeline; production scale,
  distributed curation, calibrated quality classifiers, and fixed-budget
  ADA training ablations are pending.

Sources:
- https://fasttext.cc/docs/en/language-identification.html
- https://huggingface.co/facebook/fasttext-language-identification
- https://docs.nvidia.com/nemo/curator/v26.09/curate-text/process-data/language-management/language
- https://arxiv.org/abs/2406.11794
- https://huggingface.co/docs/datasets/en/stream
