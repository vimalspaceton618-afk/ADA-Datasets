"""Run a matched-budget ADA mixture experiment on two approved releases."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import sys

from ada_datasets.core import file_digest
from ada_datasets.release import verify_release
from ada_datasets.training import TrainingRelease, is_sha256


ARMS = ("control", "treatment")


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError("experiment metadata must be an object")
    return obj


def _write_json(path: Path, result: dict) -> None:
    # Publish once so interrupted reports do not masquerade as completed runs.
    staged = path.with_suffix(".pending")
    with staged.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    staged.replace(path)


def _split_digests(manifest: dict, split: str, kind: str) -> list[str]:
    return [manifest["artifacts"][path] for path in manifest["splits"][split][kind]]


def _train_mixture_recipe(manifest: dict) -> dict:
    return {key: value for key, value in manifest["recipe"].items() if key != "token_budgets"}


def _matching_targets(releases, batch_size: int, steps: int) -> int:
    quota = batch_size * releases[0].sequence_length
    for release in releases:
        cursor = release.cursor("train")
        for _ in range(steps):
            try:
                rows, cursor = release.next_batch("train", cursor, batch_size, repeat=True)
            except StopIteration as exc:
                raise ValueError("each ablation arm must have training blocks") from exc
            quota = min(quota, sum(sum(row["loss_mask"]) for row in rows))
    return quota


def plan_experiment(
    paths: dict[str, Path],
    pins: dict[str, str],
    status: Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    device: str,
    model_profile: str,
) -> dict:
    """Verify both arms and compute an identical active-target quota per step."""
    if (steps < 1 or batch_size < 1 or set(paths) != set(ARMS) or set(pins) != set(ARMS) or
            not all(is_sha256(pin) for pin in pins.values()) or pins["control"] == pins["treatment"]):
        raise ValueError("provide two distinct pinned releases and a positive training budget")
    with ExitStack() as stack:
        releases = []
        for arm in ARMS:
            release = stack.enter_context(TrainingRelease(paths[arm], pins[arm], status_path=status))
            verify_release(paths[arm], pins[arm])
            releases.append(release)
        control, treatment = releases
        left, right = control.manifest, treatment.manifest
        if (control.tokenizer_sha256 != treatment.tokenizer_sha256 or
                control.sequence_length != treatment.sequence_length or
                left["effective_denylist_sha256"] != right["effective_denylist_sha256"] or
                _train_mixture_recipe(left) != _train_mixture_recipe(right)):
            raise ValueError("release recipes must match apart from train token budgets")
        if _split_digests(left, "train", "blocks") == _split_digests(right, "train", "blocks"):
            raise ValueError("the ablation arms contain identical training blocks")
        for split in ("validation", "test"):
            if (not left["splits"][split]["block_count"] or
                    any(_split_digests(left, split, kind) != _split_digests(right, split, kind)
                        for kind in ("documents", "blocks"))):
                raise ValueError(f"both arms require the same non-empty ordered {split} artifacts")
        quota = _matching_targets(releases, batch_size, steps)
        return {
            "schema_version": 1,
            "release_manifest_sha256": dict(pins),
            "tokenizer_sha256": control.tokenizer_sha256,
            "sequence_length": control.sequence_length,
            "batch_size": batch_size,
            "steps": steps,
            "seed": seed,
            "device": device,
            "model_profile": model_profile,
            "targets_per_step": quota,
            "training_active_targets_per_arm": quota * steps,
            "test_active_targets_per_arm": left["splits"]["test"]["active_targets"],
        }


def _valid_loss(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def run_experiment(
    trainer: Path,
    output: Path,
    paths: dict[str, Path],
    status: Path,
    plan: dict,
) -> dict:
    """Train both arms, then evaluate their completed checkpoints on test."""
    trainer = trainer.resolve(strict=True)
    if not trainer.is_file():
        raise ValueError("trainer must be a Python file")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan = {**plan, "trainer_script_sha256": file_digest(trainer)}
    _write_json(output / "ablation_plan.json", plan)
    commands, summaries = {}, {}
    pins = plan["release_manifest_sha256"]
    target_budget = plan["training_active_targets_per_arm"]
    for arm in ARMS:
        command = [
            sys.executable, str(trainer), "--output-dir", str(output / arm),
            "--curated-release", str(paths[arm]), "--curated-release-sha256", pins[arm],
            "--release-status-registry", str(status), "--max-steps", str(plan["steps"]),
            "--batch-size", str(plan["batch_size"]), "--seed", str(plan["seed"]),
            "--device", plan["device"], "--model-profile", plan["model_profile"],
            "--targets-per-step", str(plan["targets_per_step"]),
            "--max-train-tokens", str(target_budget),
        ]
        subprocess.run(command, check=True)
        summary = _json(output / arm / "run_summary.json")
        if (summary.get("status") != "COMPLETED" or summary.get("dataset_revision") != pins[arm] or
                summary.get("target_steps") != plan["steps"] or summary.get("tokens_seen_total") != target_budget or
                not _valid_loss(summary.get("latest_validation_loss"))):
            raise ValueError(f"{arm} training output does not match the ablation plan")
        commands[arm] = command
        summaries[arm] = {"validation_loss": summary["latest_validation_loss"]}

    for arm in ARMS:
        subprocess.run([*commands[arm], "--evaluate-test", "--resume"], check=True)
        test = _json(output / arm / "test_evaluation.json")
        if (test.get("release_manifest_sha256") != pins[arm] or
                test.get("active_targets") != plan["test_active_targets_per_arm"] or
                not is_sha256(test.get("checkpoint_manifest_sha256")) or not _valid_loss(test.get("test_loss"))):
            raise ValueError(f"{arm} test metrics do not match the verified experiment")
        summaries[arm].update(test_loss=test["test_loss"], checkpoint_manifest_sha256=test["checkpoint_manifest_sha256"])
    result = {
        "plan": plan,
        "results": summaries,
        "treatment_minus_control_test_loss": summaries["treatment"]["test_loss"] - summaries["control"]["test_loss"],
    }
    _write_json(output / "ablation_report.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
        parser.add_argument(f"--{arm}-sha256", required=True)
    parser.add_argument("--status-registry", type=Path, required=True)
    parser.add_argument("--trainer", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model-profile", choices=("pilot", "smoke"), default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and (args.trainer is None or args.output_dir is None):
        parser.error("--trainer and --output-dir are required for an executed experiment")
    paths = {arm: getattr(args, arm) for arm in ARMS}
    pins = {arm: getattr(args, f"{arm}_sha256") for arm in ARMS}
    try:
        result = plan_experiment(paths, pins, args.status_registry, steps=args.steps,
                                 batch_size=args.batch_size, seed=args.seed, device=args.device,
                                 model_profile=args.model_profile)
        if not args.dry_run:
            result = run_experiment(args.trainer, args.output_dir, paths, args.status_registry, result)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"ablation: {exc}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
