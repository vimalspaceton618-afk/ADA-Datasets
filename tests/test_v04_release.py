"""End-to-end coverage for the v0.4 fixture and the installed CLI."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples" / "make_release_fixture.py"
LANGUAGES = {"en", "ta", "hi"}
SPLITS = ("train", "validation", "test")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(
    importlib.util.find_spec("pyarrow") and importlib.util.find_spec("tokenizers"),
    "release extras are needed for the Parquet integration",
)
class ReleaseIntegrationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = self.root / "fixture"
        subprocess.run(
            [sys.executable, str(FIXTURE), "--output-dir", str(self.fixture)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.recipe = self.fixture / "recipe.json"

    def cli(self, *arguments, success=True):
        result = subprocess.run(
            [sys.executable, "-m", "ada_datasets.cli", *map(str, arguments)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        self.assertNotIn("fictional-contact@example.invalid", output)
        if success:
            self.assertEqual(result.returncode, 0, output)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, output)
        return output

    def release(self, name, *arguments, workers=1, success=True):
        work = self.root / f"work-{name}"
        output = self.root / name
        result = self.cli(
            "release", "--recipe", self.recipe, "--work-dir", work, "--output", output,
            "--workers", workers, *arguments, success=success,
        )
        return work, output, result

    def test_parallel_resume_and_audit_reconciliation(self):
        from pyarrow import parquet
        from ada_datasets.training import TrainingRelease

        work, resumed, prepared = self.release("resumed", "--prepare-only")
        self.assertEqual(prepared["tasks"], 2)
        self.assertFalse(resumed.exists())

        def run_task():
            return self.cli("worker", "--work-dir", work, "--task-id", 0)

        with ThreadPoolExecutor(max_workers=2) as executor:
            attempts = [executor.submit(run_task) for _ in range(2)]
            self.assertEqual(attempts[0].result(), attempts[1].result())
        self.release("resumed", workers=0, success=False)
        self.assertFalse(resumed.exists())
        self.cli("worker", "--work-dir", work, "--task-id", 1)
        self.release("resumed", workers=0)
        _, serial, _ = self.release("serial")
        _, parallel, _ = self.release("parallel", workers=2)
        manifest_sha = _sha(resumed / "manifest.json")
        manifest = json.loads((resumed / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["format"], "ada-training-release")
        self.assertEqual(manifest_sha, _sha(serial / "manifest.json"))
        self.assertEqual(manifest_sha, _sha(parallel / "manifest.json"))
        for artifact, digest in manifest["artifacts"].items():
            self.assertEqual(_sha(resumed / artifact), digest)
            self.assertEqual(_sha(serial / artifact), digest)
            self.assertEqual(_sha(parallel / artifact), digest)
        self.assertTrue(
            self.cli("verify", "--output", resumed, "--expected-manifest-sha256", manifest_sha)[
                "valid"
            ]
        )

        accepted = set()
        reader = TrainingRelease(resumed, manifest_sha)
        self.addCleanup(reader.close)
        for split in SPLITS:
            documents = [
                row
                for path in manifest["splits"][split]["documents"]
                for row in parquet.read_table(resumed / path).to_pylist()
            ]
            self.assertEqual({row["language_declared"] for row in documents}, LANGUAGES)
            self.assertEqual({row["split"] for row in documents}, {split})
            if split == "test":
                related = {
                    row["id"]: row["group_sha256"]
                    for row in documents
                    if row["id"] in ("ada-synthetic:ta-test", "ada-synthetic:ta-test-linked")
                }
                self.assertEqual(len(related), 2)
                self.assertEqual(len(set(related.values())), 1)
            for row in documents:
                self.assertNotIn(row["id"], accepted)
                self.assertNotIn("fictional-contact@example.invalid", row["text"])
                self.assertIn("synthetic fixture", row["attribution"])
                self.assertEqual((row["license"], row["rights_basis"]), ("CC0-1.0", "owned"))
                accepted.add(row["id"])
            cursor = reader.cursor(split)
            first, next_cursor = reader.next_batch(split, cursor, 2)
            self.assertEqual(reader.next_batch(split, cursor, 2), (first, next_cursor))
            self.assertEqual(len(first), 2)
            self.assertTrue(all(len(row["input_ids"]) == 33 for row in first))
            self.assertTrue(all(len(row["loss_mask"]) == 32 for row in first))
            self.assertTrue(any(any(row["loss_mask"]) for row in first))
        self.assertEqual(len(accepted), 10)
        self.assertEqual(
            len(
                accepted
                & {
                    "ada-synthetic:en-train",
                    "ada-synthetic:en-train-exact",
                    "ada-synthetic:en-train-near",
                }
            ),
            1,
        )

        audit = resumed / "audit.jsonl"
        self.assertNotIn("fictional-contact@example.invalid", audit.read_text(encoding="utf-8"))
        audit_rows = [json.loads(line) for line in audit.read_bytes().splitlines()]
        reasons = {row["reason"] for row in audit_rows}
        for reason in ("duplicate_exact", "duplicate_near", "denylisted"):
            self.assertIn(reason, reasons)
        self.assertTrue(any("contamination" in reason for reason in reasons))
        self.assertTrue(any("pii" in reason for reason in reasons))
        self.assertTrue(any("rights" in reason for reason in reasons))
        self.assertTrue(any("invalid" in reason for reason in reasons))

        source_lines = [
            (self.fixture / f"input-{number}.jsonl").read_bytes().splitlines(keepends=True)
            for number in range(2)
        ]
        self.assertEqual(len(audit_rows) + len(accepted), sum(map(len, source_lines)))
        for row in audit_rows:
            self.assertEqual(
                set(row),
                {"input_index", "line", "input_line_sha256", "source_id", "decision", "reason"},
            )
            raw = source_lines[row["input_index"]][row["line"] - 1]
            self.assertEqual(row["input_line_sha256"], sha256(raw).hexdigest())
        self.assertTrue(
            any(
                source_lines[row["input_index"]][row["line"] - 1].endswith(b"\r\n")
                for row in audit_rows
            )
        )

    def test_tamper_rejected_by_verifier_and_reader(self):
        from ada_datasets.training import TrainingRelease

        _, output, _ = self.release("tamper")
        manifest_path = output / "manifest.json"
        digest = _sha(manifest_path)
        manifest = json.loads(manifest_path.read_bytes())
        self.cli("verify", "--output", output, "--expected-manifest-sha256", "0" * 64, success=False)
        loader = TrainingRelease(output, digest)
        self.addCleanup(loader.close)
        shard = output / manifest["splits"]["train"]["blocks"][0]
        with shard.open("ab") as file:
            file.write(b"tampered bytes")
        self.cli("verify", "--output", output, "--expected-manifest-sha256", digest, success=False)
        with self.assertRaises((OSError, ValueError)):
            loader.next_batch("train", loader.cursor("train"), 1)

    def test_modified_intake_fails_without_publication(self):
        with (self.fixture / "input-0.jsonl").open("ab") as input_file:
            input_file.write(b"{}\n")
        _, output, _ = self.release("drift", success=False)
        self.assertFalse(output.exists())

    def test_token_budgets_only_limit_train_and_report_shortfalls(self):
        recipe = json.loads(self.recipe.read_bytes())
        recipe["token_budgets"] = {"ta": 1, "hi/education": 0}
        self.recipe.write_text(json.dumps(recipe), encoding="utf-8")
        _, output, _ = self.release("budgeted")
        manifest = json.loads((output / "manifest.json").read_bytes())
        composition = manifest["composition"]
        self.assertEqual(set(composition["splits"]["train"]["language"]), {"en"})
        for split in ("validation", "test"):
            self.assertEqual(set(composition["splits"][split]["language"]), LANGUAGES)
        self.assertEqual(manifest["totals"]["budget_excluded"], 2)
        self.assertEqual(
            composition["token_budgets"]["ta"],
            {"limit": 1, "used": 0, "remaining": 1, "excluded_documents": 1},
        )

    def test_successor_suppresses_exact_alias_and_quarantines_parent(self):
        from pyarrow import parquet
        from ada_datasets.training import TrainingRelease

        status = self.root / "status.json"
        self.cli("governance", "--status", status, "init")
        _, parent, parent_result = self.release("parent", "--status", status)
        parent_digest = parent_result["manifest_sha256"]
        old = TrainingRelease(parent, parent_digest, status_path=status)
        self.addCleanup(old.close)
        self.assertTrue(old.next_batch("train", old.cursor("train"), 1)[0])
        checkpoint = sha256(b"synthetic attached checkpoint").hexdigest()
        self.cli(
            "governance", "--status", status, "register-checkpoint",
            "--release-sha256", parent_digest, "--checkpoint-sha256", checkpoint,
        )
        original = next(
            json.loads(line)["text"]
            for line in (self.fixture / "input-0.jsonl").read_bytes().splitlines()
            if json.loads(line)["id"] == "en-train"
        )
        old_ids = {"ada-synthetic:en-train", "ada-synthetic:en-train-exact"}
        old_text_digest = sha256(original.encode()).hexdigest()
        denylist = self.fixture / "updated-denylist.jsonl"
        denylist.write_bytes(
            (self.fixture / "denylist.jsonl").read_bytes()
            + b'{"doc_id":"ada-synthetic:en-train-exact"}\n'
        )
        recipe = json.loads(self.recipe.read_bytes())
        recipe["denylist"] = {"path": denylist.name, "sha256": _sha(denylist)}
        self.recipe.write_text(json.dumps(recipe), encoding="utf-8")
        _, successor, verified = self.release(
            "successor", "--parent", parent, "--status", status, workers=2
        )
        with self.assertRaisesRegex(ValueError, "quarantined"):
            old.next_batch("train", old.cursor("train"), 1)
        final_digest = verified["manifest_sha256"]
        with TrainingRelease(successor, final_digest, status_path=status) as current:
            batches, _ = current.next_batch("train", current.cursor("train"), 1)
            self.assertTrue(batches)
        self.assertEqual(parent_digest, _sha(parent / "manifest.json"))
        self.assertTrue(
            self.cli("verify", "--output", successor, "--expected-manifest-sha256", final_digest)[
                "valid"
            ]
        )
        manifest = json.loads((successor / "manifest.json").read_bytes())
        self.assertEqual(manifest["lineage"]["parent_manifest_sha256"], parent_digest)
        impact_path = successor / "revocation-impact.json"
        self.assertEqual(_sha(impact_path), manifest["artifacts"][impact_path.name])
        impact = json.loads(impact_path.read_bytes())
        self.assertEqual(impact["matched_documents"], 1)
        self.assertEqual(impact["document_sha256"], [old_text_digest])
        self.assertTrue(impact["document_artifacts"])
        self.assertTrue(impact["block_artifacts"])
        for split in SPLITS:
            for name in manifest["splits"][split]["documents"]:
                table = parquet.read_table(successor / name, columns=["id", "sha256"])
                self.assertTrue(old_ids.isdisjoint(table.column("id").to_pylist()))
                self.assertNotIn(old_text_digest, table.column("sha256").to_pylist())
        registry = self.cli("governance", "--status", status, "verify")
        self.assertEqual(registry["active_releases"], [final_digest])
        self.assertEqual(registry["pending_deletions"]["release"], [parent_digest])
        self.assertEqual(registry["pending_deletions"]["checkpoint"], [checkpoint])


if __name__ == "__main__":
    unittest.main()
