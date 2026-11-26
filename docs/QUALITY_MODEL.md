# Local quality model training

Train the optional binary fastText scorer on independently reviewed keep/drop labels from source material you can use for this purpose. Run training and calibration on a private Linux host.

```sh
python -m pip install -e '.[models]'
ada-curate train-quality \
  --fit private/quality-fit.jsonl --fit-sha256 "$ADA_FIT_SHA256" \
  --calibration private/quality-calibration.jsonl --calibration-sha256 "$ADA_CALIBRATION_SHA256" \
  --model-license 'reviewed-model-license' --output private/quality-v1
```

Get each SHA-256 from the exact JSONL file using `sha256sum`. Each record has `text`, `language` (`en`, `ta` or `hi`), Boolean `should_keep`, and an optional opaque `family_id`. Group related documents before splitting the fit, calibration and final evaluation sets. Include both decisions for every language in both files. Calibration requires at least five labels of each class per language by default. The trainer rejects repeated normalized inputs and any supplied family shared across fit and calibration.

The command saves `quality.bin`, a recipe fragment `quality-model.json`, and a report of per-language label counts and calibration metrics. Use the fragment as the release recipe's `quality_model` value. Set its `path` relative to the recipe file. The scorer uses one fastText training thread and a fixed seed. The report records the library version, training parameters and both input hashes.

Thresholds maximize F1 independently for each language, using precision and then the higher threshold to break ties. Measure final performance on independent held-out labels using `ada-curate assess-release`. Calibrating the decision threshold on a labeled sample leaves the underlying fastText score uncalibrated as a probability.

Keep the model private. FastText files include learned vocabulary derived from input. Review the training labels and intended model use before fitting, and use a representative independent test set before adopting a trained filter.
