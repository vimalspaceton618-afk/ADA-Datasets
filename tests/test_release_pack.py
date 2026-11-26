"""Native Parquet release tests using the real ByteLevel tokenizer and Arrow."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ada_datasets.release_pack import pack_release, verify_pack
from ada_datasets.training import SPECIAL_TOKENS, TrainingRelease


@unittest.skipUnless(
    importlib.util.find_spec("pyarrow") and importlib.util.find_spec("tokenizers"),
    "optional release dependencies are required",
)
class ReleasePackTests(unittest.TestCase):
    def setUp(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.tokenizer_path = self.root / "tokenizer.json"
        self.tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.train_from_iterator(
            ["one two three", "varied writing patterns", "வணக்கம்", "नमस्ते"],
            trainers.BpeTrainer(vocab_size=260, show_progress=False, special_tokens=list(SPECIAL_TOKENS),
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
        )
        self.tokenizer.save(str(self.tokenizer_path))
        self.tokenizer.encode_special_tokens = True
        self.tokenizer_sha = sha256(self.tokenizer_path.read_bytes()).hexdigest()

    def row(self, label, text, split, language="en", *, group=None):
        return {
            "id": f"fixture:{label}", "text": text, "sha256": sha256(text.encode()).hexdigest(),
            "source_id": "fixture", "family_id": label, "group_sha256": group or sha256(label.encode()).hexdigest(),
            "split": split, "language_declared": language, "domain_declared": "general",
            "quality_score": 1.0, "license": "CC0-1.0", "rights_basis": "owned", "attribution": "Fixture",
        }

    def build(self, rows, *, path="release", width=4, shard_rows=2, budgets=None):
        output = self.root / path
        output.mkdir()
        metadata = pack_release(
            output, {split: iter(rows.get(split, [])) for split in ("train", "validation", "test")},
            tokenizer_path=self.tokenizer_path, tokenizer_sha256=self.tokenizer_sha,
            sequence_length=width, shard_rows=shard_rows, token_budgets=budgets or {},
        )
        manifest = {"schema_version": 2, "format": "ada-training-release", **metadata}
        data = json.dumps(manifest, sort_keys=True).encode()
        (output / "manifest.json").write_bytes(data)
        return output, manifest, sha256(data).hexdigest()

    def test_masks_and_budgets_count_every_content_token_and_eos(self):
        texts = ["one <|pad|> two", "வணக்கம்", "नमस्ते"]
        expected = [self.tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
        rows = {
            "train": [self.row("a", texts[0], "train"), self.row("b", texts[1], "train", "ta"),
                      self.row("exclude", "longer text for excluded document", "train")],
            "test": [self.row("c", texts[2], "test", "hi")],
        }
        output, manifest, digest = self.build(rows, budgets={"en": len(expected[0]) + 1, "en/general": 100, "hi": 0})
        verify_pack(output, manifest)
        self.assertEqual(manifest["composition"]["budget_excluded_documents"], 1)
        self.assertEqual(manifest["composition"]["token_budgets"]["en"]["remaining"], 0)
        with TrainingRelease(output, digest) as release:
            eos = release.special_tokens["<|eos|>"]
            for split, expected_tokens in (
                ("train", expected[0] + [eos] + expected[1] + [eos]),
                ("test", expected[2] + [eos]),
            ):
                blocks, cursor = release.next_batch(split, release.cursor(split), 100)
                target_ids = [i for block in blocks for i, kept in zip(block["input_ids"][1:], block["loss_mask"]) if kept]
                self.assertEqual(target_ids, expected_tokens)
                with self.assertRaises(StopIteration):
                    release.next_batch(split, cursor, 1)
            with self.assertRaises(StopIteration):
                release.next_batch("validation", release.cursor("validation"), 1, repeat=True)

    def test_resume_crosses_row_groups_shards_and_epochs(self):
        rows = {"train": [self.row("long", "one two three " * 120, "train")]}
        output, manifest, digest = self.build(rows, shard_rows=300)
        count = manifest["splits"]["train"]["block_count"]
        self.assertGreater(count, 300)
        with TrainingRelease(output, digest) as release:
            part, saved = release.next_batch("train", release.cursor("train"), 256)
            self.assertEqual(saved, {"shard_index": 0, "row_group": 1, "row": 0, "epoch": 0})
            continuous, _ = release.next_batch("train", release.cursor("train"), count + 4, repeat=True)
        with TrainingRelease(output, digest) as resumed:
            rest, final = resumed.next_batch("train", saved, count - len(part) + 4, repeat=True)
            self.assertEqual(part + rest, continuous)
            self.assertEqual(final, {"shard_index": 0, "row_group": 0, "row": 4, "epoch": 1})

    def test_tamper_detected_before_loading_the_affected_shard(self):
        rows = {"train": [self.row("a", "one two three", "train")]}
        output, manifest, digest = self.build(rows, shard_rows=1)
        _, bad = manifest["splits"]["train"]["blocks"][:2]
        with (output / bad).open("ab") as stream:
            stream.write(b"changed")
        with TrainingRelease(output, digest) as release:
            _, cursor = release.next_batch("train", release.cursor("train"), 1)
            with self.assertRaisesRegex(ValueError, "artifact digest"):
                release.next_batch("train", cursor, 1)
        with self.assertRaisesRegex(ValueError, "artifact digest"):
            verify_pack(output, manifest)

    def test_deep_verification_rejects_split_group_leakage(self):
        group = "a" * 64
        rows = {"train": [self.row("a", "one", "train", group=group)],
                "test": [self.row("b", "three", "test", group=group)]}
        output, manifest, _ = self.build(rows)
        with self.assertRaisesRegex(ValueError, "group spans"):
            verify_pack(output, manifest)
