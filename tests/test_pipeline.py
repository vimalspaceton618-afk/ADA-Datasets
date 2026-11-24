"""Offline integration and adversarial tests of ADA Provenance Lattice v0.1."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ada_datasets.cli import main, validate_output
from ada_datasets.core import (
    CurationConfig, NearIndex, canonical_text, choose_split, clean_url,
    curate_dataset, fingerprint, has_pii_or_secret, jaccard, load_policy,
    score_text, shingles,
)

A = (
    "A robust experiment begins with a scientific question and a careful plan. "
    "Researchers select instruments, collect measurements in consistent units, "
    "describe assumptions, analyze uncertainty, and compare independent results. "
    "They also record every important decision so another scientist can reproduce "
    "the procedure and evaluate the published conclusions without guessing."
)
B = (
    "Systems programmers build trustworthy tools by specifying behavior before "
    "modifying complex implementations. Clear interfaces, deterministic unit tests, "
    "structured error reports, and reproducible integration checks help reviewers "
    "identify hidden assumptions and prevent regressions in production systems. "
    "Measured performance matters, but correctness and security remain essential."
)


class ADADataCurationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = self.root / "policy.json"
        self.policy.write_text(json.dumps({
            "schema_version": 1,
            "sources": [
                {"source_id": "author", "license": "CC0-1.0",
                 "rights_basis": "owned", "training_allowed": True},
                {"source_id": "unapproved", "license": "CC0-1.0",
                 "rights_basis": "licensed", "training_allowed": False}
            ]
        }))
        self.infile = self.root / "input.jsonl"

    def row(self, id, text=A, **extra):
        data = {"id": id, "text": text, "source_id": "author",
                "license": "CC0-1.0", "rights_basis": "owned",
                "language": "en"}
        data.update(extra)
        return data

    def write(self, rows):
        self.infile.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")

    def run_pipeline(self, output="curated", **options):
        return curate_dataset(self.infile, self.policy, self.root / output, **options)

    def emitted(self, directory="curated"):
        output = []
        for label in ("train", "validation", "test"):
            path = self.root / directory / f"{label}.jsonl"
            output.extend(json.loads(line) for line in path.read_text().splitlines())
        return output

    def rejected(self, directory="curated"):
        path = self.root / directory / "audit.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_approved_source_yields_hashed_verified_manifest(self):
        self.write([self.row("a", url="https://example.org/read?a=secret#fragment"),
                    self.row("b", B)])
        result = self.run_pipeline()
        self.assertEqual(result["totals"]["accepted"], 2)
        verified = validate_output(self.root / "curated")
        self.assertTrue(verified["valid"])
        data = self.emitted()
        self.assertEqual(len(data), 2)
        a = next(x for x in data if x["id"] == "author:a")
        self.assertEqual(a["source_url"], "https://example.org/read")
        self.assertEqual(a["rights_basis"], "owned")
        self.assertEqual(a["sha256"], fingerprint(canonical_text(A)))
        self.assertNotIn("secret", a["source_url"])

    def test_fail_closed_registry_and_license_mismatch(self):
        self.write([
            self.row("a", source_id="unapproved", rights_basis="licensed"),
            self.row("b", license="CC-BY-4.0"),
            self.row("c", source_id="nonexistent"),
        ])
        result = self.run_pipeline()
        self.assertEqual(result["totals"].get("accepted", 0), 0)
        self.assertEqual(result["totals"]["rejected_source_not_approved"], 2)
        self.assertEqual(result["totals"]["rejected_provenance_mismatch"], 1)

    def test_pii_secret_and_explicit_opt_out(self):
        self.write([
            self.row("email", A + " Please contact jane.doe@example.org soon."),
            self.row("phone", A + " Contact +91 98765 43210."),
            self.row("key", A + " -----BEGIN PRIVATE KEY-----"),
            self.row("opt", A, opt_out=True),
            self.row("notrain", A, do_not_train=True),
            self.row("card", A + " This test card is 4111111111111111."),
            self.row("malformed-consent", A, opt_out="false"),
        ])
        result = self.run_pipeline()
        self.assertEqual(result["totals"].get("accepted", 0), 0)
        self.assertEqual(result["totals"]["rejected_pii_or_secret"], 4)
        self.assertEqual(result["totals"]["rejected_record_opt_out"], 2)
        self.assertEqual(result["totals"]["rejected_invalid_optout_flag"], 1)
        audit = (self.root / "curated/audit.jsonl").read_text()
        self.assertNotIn("jane.doe", audit)
        self.assertNotIn("PRIVATE KEY", audit)
        self.assertNotIn("98765", audit)
        self.assertNotIn("4111111111111111", audit)

    def test_exact_and_shingle_identical_near_duplicate(self):
        text_variant = A.replace("scientific question", "scientific, question")
        self.assertNotEqual(fingerprint(A), fingerprint(text_variant))
        self.assertEqual(shingles(A), shingles(text_variant))
        self.write([self.row("one", A), self.row("two", A),
                    self.row("three", text_variant)])
        result = self.run_pipeline()
        self.assertEqual(result["totals"]["accepted"], 1)
        self.assertEqual(result["totals"]["rejected_duplicate_exact"], 1)
        self.assertEqual(result["totals"]["rejected_duplicate_near"], 1)

    def test_benchmark_and_denylist_exclusion(self):
        benchmark = self.root / "heldout.jsonl"
        benchmark.write_text(json.dumps({"text": A}) + "\n")
        denylist = self.root / "deleted.jsonl"
        denylist.write_text(json.dumps({"doc_id": "author:b"}) + "\n")
        self.write([self.row("a", A), self.row("b", B)])
        result = self.run_pipeline(holdout_path=benchmark, denylist_path=denylist)
        self.assertEqual(result["totals"]["rejected_benchmark_contamination"], 1)
        self.assertEqual(result["totals"]["rejected_denylisted"], 1)
        self.assertFalse(self.emitted())

    def test_reproducibility_same_input_and_seed(self):
        self.write([self.row("a", A), self.row("b", B)])
        first = self.run_pipeline("first")
        second = self.run_pipeline("second")
        self.assertEqual(first, second)
        for name in ("manifest.json", "train.jsonl", "validation.jsonl",
                     "test.jsonl", "audit.jsonl"):
            self.assertEqual((self.root / "first" / name).read_bytes(),
                             (self.root / "second" / name).read_bytes())

    def test_family_exclusivity(self):
        self.write([self.row("a", A, family_id="family-x"),
                    self.row("b", B, family_id="family-x")])
        self.run_pipeline()
        self.assertEqual(len({r["split"] for r in self.emitted()}), 1)

    def test_corruption_detection(self):
        self.write([self.row("a", A)])
        self.run_pipeline()
        file = next((self.root / "curated" / f"{s}.jsonl"
                    for s in ("train", "validation", "test")
                    if (self.root / "curated" / f"{s}.jsonl").stat().st_size))
        file.write_text(file.read_text() + "corruption")
        with self.assertRaisesRegex(ValueError, "corrupt output"):
            validate_output(self.root / "curated")

    def test_no_implicit_overwrite(self):
        self.write([self.row("a")])
        self.run_pipeline()
        with self.assertRaises(FileExistsError):
            self.run_pipeline()
        self.assertEqual(main(["verify", "--output",
                               str(self.root / "curated")]), 0)

    def test_quality_length_and_language_rejection(self):
        self.write([self.row("tiny", "x"),
                    self.row("missinglang", B, language="xx"),
                    self.row("duplicate-id", B),
                    self.row("duplicate-id", A)])
        result = self.run_pipeline()
        self.assertEqual(result["totals"]["rejected_too_short"], 1)
        self.assertEqual(result["totals"]["rejected_unsupported_or_undeclared_language"], 1)
        self.assertEqual(result["totals"]["rejected_duplicate_document_id"], 1)

    def test_invalid_input_metadata_does_not_crash_pipeline(self):
        self.infile.write_text("{bad json\n" + json.dumps(
            self.row("valid", A)) + "\n")
        result = self.run_pipeline()
        self.assertEqual(result["totals"]["rejected_invalid_json_or_metadata"], 1)
        self.assertEqual(result["totals"]["accepted"], 1)

    def test_rejected_unregistered_source_id_is_not_copied_to_audit(self):
        self.write([self.row("x", source_id="private-password-in-source-name")])
        self.run_pipeline()
        audit = (self.root / "curated/audit.jsonl").read_text()
        self.assertNotIn("private-password-in-source-name", audit)

    def test_hf_adapter_requires_approval_before_loading_dependencies(self):
        from types import SimpleNamespace
        from ada_datasets.hf_adapter import adapt_hf
        args = SimpleNamespace(
            acknowledge_source_rights=False,
        )
        with self.assertRaisesRegex(ValueError, "acknowledge-source-rights"):
            adapt_hf(args)

    def test_exact_split_reproducibility(self):
        self.assertEqual(choose_split("source:family", 2026),
                         choose_split("source:family", 2026))
        self.assertIn(choose_split("source:family", 2026),
                      ("train", "validation", "test"))

    def test_url_rejects_nonhttp_embedded_credentials(self):
        for url in ("file:///etc/passwd", "https://admin:pwd@private.net/path",
                    "http://"):
            with self.assertRaises(ValueError):
                clean_url(url)

    def test_minimum_pii_patterns(self):
        self.assertTrue(has_pii_or_secret("hello world abc@example.org"))
        self.assertTrue(has_pii_or_secret("my mobile +91 98765 43210"))
        self.assertFalse(has_pii_or_secret(A))

    def test_jaccard_and_quality_checks(self):
        self.assertEqual(jaccard(frozenset(), frozenset()), 1)
        self.assertGreaterEqual(score_text(A)[0], 0.65)
        self.assertLess(score_text("repeat repeat repeat repeat")[0],
                        score_text(A)[0])

    def test_fail_closed_invalid_policy(self):
        self.policy.write_text(json.dumps({"schema_version": 1, "sources": [
            {"source_id": "a", "training_allowed": True,
             "license": "CC0-1.0", "rights_basis": "owned"},
            {"source_id": "a", "training_allowed": True,
             "license": "CC0-1.0", "rights_basis": "owned"}
        ]}))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_policy(self.policy)

    def test_reject_bad_config(self):
        with self.assertRaises(ValueError):
            CurationConfig(min_quality=2)
        with self.assertRaises(ValueError):
            CurationConfig(seed=-1)

    def test_bounded_holdout(self):
        self.write([self.row("a")])
        holdout = self.root / "holdout.jsonl"
        holdout.write_text('{"text":"hello"}\n{"text":"world"}\n')
        with self.assertRaisesRegex(ValueError, "cap exceeded"):
            self.run_pipeline(holdout_path=holdout,
                              config=CurationConfig(max_holdout_records=1))


if __name__ == "__main__":
    unittest.main()
