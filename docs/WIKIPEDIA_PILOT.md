# Wikipedia candidate review and ADA GPU pilot

## Candidate approval packet

The first proposed private pilot uses three configurations from `wikimedia/wikipedia`, all at Hub revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` and Hub split `train`. ADA constructs its own train, validation, and test groups. The Hub split name applies only to upstream access. [Pinned dataset card](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa) · [Hub revision metadata](https://huggingface.co/api/datasets/wikimedia/wikipedia?expand%5B%5D=sha)

| Configuration | Language | Source ID | Initial row cap |
| --- | --- | --- | ---: |
| `20231101.en` | English | `wikipedia-20231101-en` | 20,000 |
| `20231101.ta` | Tamil | `wikipedia-20231101-ta` | 20,000 |
| `20231101.hi` | Hindi | `wikipedia-20231101-hi` | 20,000 |

**Approval status is pending.** The pinned card describes a cleaned Wikipedia dump and references CC BY-SA 3.0 and GFDL. Wikimedia's current Terms of Use describe CC BY-SA 4.0 and GFDL for text contributions, imported-content exceptions, attribution obligations, and rules for modifications. The reviewer must reconcile the pinned snapshot's applicable terms and exceptions, preserve per-article source URLs, and record the intended uses and attribution approach. Proposed scope is internal research-only pilot training with restricted storage and no planned dataset or checkpoint publication. [Wikimedia Terms of Use, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [Wikipedia dump licensing](https://dumps.wikimedia.org/legal.html)

Before ingesting, record the authorized reviewer, review date, digest of private evidence, scope and allowed uses, license and attribution decision, retention date, and withdrawal contact. Bind each approved source policy entry to the dataset, configuration, split, and exact revision above. Spot-check page URLs and notices. Exclude third-party material whose terms cannot be established. Keep data exports, review evidence, human labels, model files, and checkpoints in the designated private environment. Stop if the recorded source, model, and infrastructure approvals are incomplete.

## Intake and independent measurement

1. After approval, use `ada-curate adapt-hf` once per table row with `--dataset-id wikimedia/wikipedia`, `--revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, `--split train`, `--max-rows 20000`, the matching `--config`, `--source-id`, and `--language`, and `--acknowledge-source-rights`. Set `--sources` to the approved private registry and `--output` to a private JSONL path. Record actual row counts and input digests. Keep the language/source assignments and input order fixed across both arms.
2. Sample across languages, sources, and content lengths, including rejected records. Use phase-separated train, calibration, and held-out sets for human annotations. Group near-related records before splitting the annotations. Record the instructions, independent reviewers, disagreement rate, adjudication, and the count of labeled positives and negatives per slice. Build a private upstream holdout list of external benchmark passages and labels reserved for measuring intake gates.
3. Use `ada-curate train-quality` to fit the authorized local fastText quality model and freeze per-language keep thresholds. Pin model bytes, package version, training configuration, source-label artifact, and approved weights license. Record the original fastText LID model digest and approved license, and choose its confidence threshold using calibration samples. Run `ada-curate evaluate-lid` and `ada-curate assess-release` against the held-out annotations. Report precision, recall, false-positive sampling, label denominators, confidence intervals, and per-language results to the reviewer. [Quality model training](QUALITY_MODEL.md) · [Language evaluation methodology](LANGUAGE_AND_DATA_EVALUATION.md)

## Two-arm design

Use a single existing private GPU worker. Build the control and treatment from the same approved corpus with the same permissions, protected holdouts, tokenizer, language and quality gates, deduplication, and family-split seed. Train the tokenizer using the approved common training split only. Verify that the validation and test shard digests are identical across the two arms.

- **Control** uses the common eligible training pool with the natural source mixture. Use an empty `token_budgets` mapping.
- **Treatment** uses equal per-language training token budgets with keys `en`, `ta`, and `hi`. Set each cap to the smallest active-target total across the three languages observed in the control's train composition. Check actual emitted totals because whole-document selection can leave shortfalls. This comparison estimates the effect of the mixture under a fixed curation policy.

Use model profile `pilot`, sequence length 256, batch size 4, seeds 2026, 2027, and 2028, and 500 optimizer steps per arm and seed. Use `--dry-run` to preflight the two releases and compute a common active-target quota. Fix the model, paired initialization, learning-rate schedule, tokenizer, software versions, and device. Store the preflight plan and planned primary validation metric before training.

```sh
for seed in 2026 2027 2028; do
  python examples/run_ada_ablation.py --trainer "$ada_checkout/scripts/ada_pretrain_fineweb_edu.py" \
    --control "$control_release" --control-sha256 "$control_sha" \
    --treatment "$treatment_release" --treatment-sha256 "$treatment_sha" \
    --status-registry "$status_path" --output-dir "$pilot_root/ablation-$seed" \
    --seed "$seed" --device cuda --model-profile pilot --steps 500 --batch-size 4 || break
done
```

The runner verifies both shared held-out splits before training, then runs reserved-split evaluation from completed checkpoints. Finish the predeclared matrix before inspecting test outputs. Read each pair's `ablation_report.json` and its versioned checkpoints for the final report.

## Evidence and report

Collect per-run release and tokenizer digests, git revisions and package versions, seed, initial checkpoint identity, GPU type, update count, exact active targets, throughput, peak memory, validation and final test results, and any exclusions. Report paired validation-loss differences by seed, their mean and uncertainty, the realized training mixture, and per-language intake coverage. Use the same validation and test artifacts for both arms and stop on any mismatch. Record incomplete approvals, unavailable labels, low sample counts, and resume discrepancies as explicit open release gates. Publish aggregate results only after the owner reviews what the approval permits.
