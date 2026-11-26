"""Exercise the paired launch contract without private or GPU data."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ada_datasets.release import run_release
from ada_datasets.release_governance import initialize_registry
from examples.make_release_fixture import make_fixture
from examples.run_ada_ablation import plan_experiment, run_experiment


@unittest.skipUnless(
    importlib.util.find_spec("pyarrow") and importlib.util.find_spec("tokenizers"),
    "optional release dependencies are required",
)
class AblationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.status = self.root / "status.json"
        initialize_registry(self.status)
        self.fixture = make_fixture(self.root / "fixture")
        self.control_recipe = json.loads(self.fixture.read_text())
        self.paths = {arm: self.root / arm for arm in ("control", "treatment")}
        self.pins = {
            "control": run_release(self.fixture, self.root / "work-control", self.paths["control"],
                                   status=self.status)["manifest_sha256"],
            "treatment": self.build_treatment({"token_budgets": {"en": 0}}),
        }

    def build_treatment(self, overrides, *, name="treatment"):
        recipe = self.fixture.with_name(f"{name}.json")
        recipe.write_text(json.dumps({**self.control_recipe, **overrides}))
        return run_release(recipe, self.root / f"work-{name}", self.root / name,
                           status=self.status)["manifest_sha256"]

    def plan(self, paths=None, pins=None):
        return plan_experiment(paths or self.paths, pins or self.pins, self.status,
                               steps=3, batch_size=2, seed=2026, device="cpu", model_profile="smoke")

    def test_constrains_active_targets_and_evaluates_only_after_both_train(self):
        plan = self.plan()
        self.assertGreater(plan["targets_per_step"], 0)
        self.assertEqual(plan["training_active_targets_per_arm"], plan["targets_per_step"] * 3)
        launch_order = []

        def trainer_run(command, *, check):
            self.assertTrue(check)
            directory = Path(command[command.index("--output-dir") + 1])
            directory.mkdir(exist_ok=True)
            arm = directory.name
            self.assertEqual(int(command[command.index("--targets-per-step") + 1]), plan["targets_per_step"])
            self.assertEqual(int(command[command.index("--max-train-tokens") + 1]),
                             plan["training_active_targets_per_arm"])
            if "--evaluate-test" in command:
                self.assertIn("--resume", command)
                launch_order.append((arm, "test"))
                (directory / "test_evaluation.json").write_text(json.dumps({
                    "release_manifest_sha256": self.pins[arm],
                    "checkpoint_manifest_sha256": sha256(arm.encode()).hexdigest(),
                    "active_targets": plan["test_active_targets_per_arm"], "test_loss": 3.0,
                }))
            else:
                launch_order.append((arm, "train"))
                (directory / "run_summary.json").write_text(json.dumps({
                    "status": "COMPLETED", "dataset_revision": self.pins[arm], "target_steps": 3,
                    "tokens_seen_total": plan["training_active_targets_per_arm"], "latest_validation_loss": 3.1,
                    "internal_path": "private details that must stay out of the final report",
                }))

        trainer = self.root / "trainer.py"
        trainer.write_text("# launcher test mock\n")
        output = self.root / "ablation"
        with patch("examples.run_ada_ablation.subprocess.run", side_effect=trainer_run):
            result = run_experiment(trainer, output, self.paths, self.status, plan)
        self.assertEqual(launch_order, [("control", "train"), ("treatment", "train"),
                                        ("control", "test"), ("treatment", "test")])
        self.assertEqual(result["treatment_minus_control_test_loss"], 0)
        report = (output / "ablation_report.json").read_text()
        self.assertNotIn("internal_path", report)
        self.assertNotIn(str(self.root), report)

    def test_equal_training_and_changed_held_out_recipe_fail_closed(self):
        for name, changes, reason in (
            ("no-change", {"token_budgets": {"en": 1000000}}, "identical training blocks"),
            ("width-change", {"sequence_length": 16}, "release recipes must match"),
        ):
            with self.subTest(name=name):
                pin = self.build_treatment(changes, name=name)
                with self.assertRaisesRegex(ValueError, reason):
                    self.plan({**self.paths, "treatment": self.root / name}, {**self.pins, "treatment": pin})
