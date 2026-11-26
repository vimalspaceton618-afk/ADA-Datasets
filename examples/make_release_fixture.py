"""Generate an original, entirely synthetic fixture for the release pipeline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ada_datasets.core import choose_split


SOURCE = "ada-synthetic"
SEED = 2026
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
SPLITS = ("train", "validation", "test")
LANGUAGES = ("en", "ta", "hi")

PASSAGES = {
    "en": (
        (
            "A traveling library leaves its depot shortly after sunrise. Volunteers arrange books "
            "in crates ordered by village and reading level. At each stop they check returns, "
            "record requests, and reserve places for new readers. A weatherproof ledger tracks the "
            "distance of every journey so the team can plan maintenance before the busiest months. "
            "Friends sometimes leave notes suggesting stories for the next visit."
        ),
        (
            "Salt crystals grow slowly when a shallow dish of brine is left near a warm window. "
            "Students mark the starting water level and weigh the vessel each afternoon. They "
            "compare the shape of the grains under a lens and wait until two identical readings "
            "show the water has gone. A second dish kept in shade helps explain why temperature "
            "can change the pace of evaporation."
        ),
        (
            "A carpenter restores a gate beside a quiet orchard. She lifts the old latch, checks "
            "the distance between the hinges, and removes loose fibers with a narrow brush. After "
            "testing several scrap boards, she chooses a smooth finish that sheds rain. The gate "
            "swings more freely when the frame is level and each hinge has room to move."
        ),
    ),
    "ta": (
        (
            "ஊரின் சிறிய நூலகத்தில் மாணவர்கள் வாரந்தோறும் கூடினர். பழைய வரைபடங்களை மேசையில் "
            "விரித்து ஆற்றின் பாதையையும் அருகிலுள்ள மலைகளையும் கண்டறிந்தனர். அடுத்த கூட்டத்தில் "
            "நடந்த பயணங்களைப் பற்றி எழுதவும் ஒவ்வொரு இடத்திற்கும் தனித்த குறியீடு அமைக்கவும் "
            "ஆசிரியர் கேட்டார். குழுவினர் தங்கள் குறிப்புகளைப் பகிர்ந்து புதிய வரைபடம் வரைந்தனர்."
        ),
        (
            "கடற்கரையில் மீனவர்கள் காலையில் காற்றின் திசையைப் பார்த்தனர். மேகங்கள் நகரும் "
            "வேகத்தையும் அலைகளின் உயரத்தையும் ஒரு குறிப்பேட்டில் பதிவு செய்தனர். பல நாட்களின் "
            "பதிவுகளை ஒப்பிட்டு பாதுகாப்பான பயண நேரத்தைத் தேர்ந்தெடுத்தனர். இளம் பயிற்சியாளர்கள் "
            "இந்த முறையைக் கற்றுக்கொண்டு கரையில் இருக்கும் நண்பர்களுடன் தகவலைப் பகிர்ந்தனர்."
        ),
        (
            "பள்ளித் தோட்டத்தில் விதைகள் வெவ்வேறு இடங்களில் நடப்பட்டன. சில செடிகளுக்கு நேரடியாகச் "
            "சூரிய ஒளி கிடைத்தது. மற்ற செடிகள் மரத்தின் நிழலில் வளர்ந்தன. குழந்தைகள் மண்ணின் "
            "ஈரத்தையும் செடிகளின் உயரத்தையும் அளந்தனர். வாரம் முடிந்தபோது எந்தச் சூழலில் இலைகள் "
            "விரைவாக வளர்ந்தன என்பதை அவர்கள் விவாதித்தனர்."
        ),
    ),
    "hi": (
        (
            "गाँव के कारीगरों ने मिट्टी के बर्तन बनाने की कार्यशाला शुरू की। पहले उन्होंने मिट्टी "
            "से छोटे पत्थर अलग किए और पानी मिलाकर उसे नरम किया। चाक को धीरे घुमाते हुए बच्चों ने "
            "कटोरियाँ बनाईं। धूप में सूखने के बाद उन्हें भट्ठी में रखा गया। अगले दिन सभी ने आकार "
            "और रंग में आए बदलावों को अपनी पुस्तिका में लिखा।"
        ),
        (
            "पहाड़ी रास्ते पर चलते हुए एक समूह ने अलग प्रकार की चट्टानें देखीं। उन्होंने हर नमूने "
            "के पास की मिट्टी और पौधों का वर्णन किया। नीचे लौटकर समूह ने अपने अवलोकन एक तालिका में "
            "रखे और उन स्थानों की तुलना की जहाँ वर्षा का पानी अधिक देर ठहरा। इस अभ्यास से उन्हें "
            "ढलान और जल निकासी के बीच संबंध समझ आया।"
        ),
        (
            "मोहल्ले के पुराने रंगमंच में कलाकारों ने नया नाटक तैयार किया। मंच के पीछे रोशनी और "
            "पर्दे सँभालने वाले लोग अलग समय पर पहुँचे। अभ्यास के बाद निर्देशक ने संवाद बोलने की "
            "गति पर चर्चा की। दर्शकों के लिए बैठने की व्यवस्था करते समय दल ने निकास के रास्तों और "
            "गलियारों को खुला रखा।"
        ),
    ),
}

HOLDOUT = (
    "A brass pendulum taps a suspended reed while a charcoal trace marks "
    "every return along the rotating drum. The protected observation belongs "
    "to the fixture benchmark and cannot appear in the training collection."
)
PRIVATE_MARKER = "fictional-contact@example.invalid"


def _pin(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256(path.read_bytes()).hexdigest()}


def _line(data: dict, newline: bytes = b"\n") -> bytes:
    return json.dumps(data, ensure_ascii=False, sort_keys=True).encode() + newline


def _candidate(identifier: str, text: str, language: str = "en", **fields) -> dict:
    return {
        "id": identifier,
        "source_id": SOURCE,
        "license": "CC0-1.0",
        "rights_basis": "owned",
        "text": text,
        "language": language,
        "domain": "education",
        **fields,
    }


def _family_for(split: str, language: str) -> str:
    for counter in range(10_000):
        family = f"{language}-{split}-{counter}"
        family_key = json.dumps([SOURCE, family], ensure_ascii=False, separators=(",", ":"))
        if choose_split(sha256(family_key.encode()).hexdigest(), SEED) == split:
            return family
    raise RuntimeError(f"no synthetic family found for {split}")


def make_fixture(output: Path, *, sequence_length: int = 32) -> Path:
    """Write pinned policy, two input shards, and a training-only BPE tokenizer."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    output.mkdir(parents=True, exist_ok=False)
    docs = [[], []]
    train_texts = []
    english_train = None
    for language in LANGUAGES:
        for index, (split, text) in enumerate(zip(SPLITS, PASSAGES[language])):
            row = _candidate(
                f"{language}-{split}", text, language, family_id=_family_for(split, language)
            )
            docs[index % len(docs)].append(_line(row))
            if split == "train":
                train_texts.append(text)
                if language == "en":
                    english_train = row
    assert english_train is not None
    docs[1].append(
        _line(
            _candidate(
                "ta-test-linked",
                "காலைப் பேருந்து நிறுத்தத்தில் பயணிகள் வரிசையாக நின்றனர். "
                "புதிய வழித்தடம் பற்றிய அறிவிப்பைப் படித்த ஒருவர் அருகில் இருந்தவர்களுக்கு விளக்கினார். "
                "சிலர் அடுத்த ஊரில் நடைபெறும் சந்தைக்குச் சென்றனர். "
                "மழை தொடங்கியபோது ஓட்டுநர் அனைவரும் பாதுகாப்பாக ஏறியதை உறுதி செய்தார்.",
                "ta",
                family_id=_family_for("test", "ta"),
            )
        )
    )

    # The cross-shard duplicates share the source family so the train-only
    # tokenizer corpus is training-only regardless of representative ordering.
    docs[1].extend(
        [
            _line({**english_train, "id": "en-train-exact"}),
            _line(
                {
                    **english_train,
                    "id": "en-train-near",
                    "text": english_train["text"] + " Occasionally.",
                }
            ),
            _line(_candidate("private-contact", f"{PASSAGES['en'][2]} {PRIVATE_MARKER}"), b"\r\n"),
            b'{"invalid":"\xff"}\r\n',
            _line(
                _candidate(
                    "benchmark-excerpt",
                    "Technicians prepared a museum label for a temporary collection. "
                    + HOLDOUT
                    + " Their workshop includes diagrams, tools, pigments, and reference books "
                    "for visiting apprentices who prepare a worksheet after every lesson.",
                )
            ),
            _line(_candidate("withdrawn", PASSAGES["en"][2] + " Withdrawn example.")),
            _line(_candidate("unknown-rights", PASSAGES["en"][2], source_id="unapproved")),
            _line(_candidate("past-retention", PASSAGES["en"][2], source_id="expired")),
        ]
    )

    inputs = []
    for number, rows in enumerate(docs):
        path = output / f"input-{number}.jsonl"
        path.write_bytes(b"".join(rows))
        inputs.append(_pin(path))

    holdout = output / "holdout.jsonl"
    holdout.write_bytes(_line({"text": HOLDOUT}))
    denylist = output / "denylist.jsonl"
    denylist.write_bytes(_line({"doc_id": f"{SOURCE}:withdrawn"}))
    source = {
        "source_id": SOURCE,
        "license": "CC0-1.0",
        "rights_basis": "owned",
        "training_allowed": True,
        "review": {
            "reviewed_by": "synthetic-fixture",
            "reviewed_at": "2026-01-01",
            "evidence_sha256": sha256(b"original synthetic fixture").hexdigest(),
        },
        "allowed_uses": ["pretraining"],
        "commercial_allowed": True,
        "redistribution_allowed": True,
        "retention_until": None,
        "attribution": "ADA-Datasets original synthetic fixture",
        "opt_out_contact": "https://example.invalid/fixture",
    }
    policy = output / "sources.json"
    policy.write_bytes(
        _line(
            {
                "schema_version": 2,
                "sources": [
                    source,
                    {**source, "source_id": "expired", "retention_until": "2026-01-02"},
                ],
            }
        )
    )

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        train_texts,
        trainers.BpeTrainer(
            vocab_size=512,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    recipe = output / "recipe.json"
    recipe.write_bytes(
        _line(
            {
                "schema_version": 1,
                "inputs": inputs,
                "policy": _pin(policy),
                "holdout": _pin(holdout),
                "denylist": _pin(denylist),
                "tokenizer": _pin(tokenizer_path),
                "sequence_length": sequence_length,
                "shard_rows": 8,
                "token_budgets": {},
                "curation": {
                    "as_of_date": "2026-01-03",
                    "seed": SEED,
                    "min_chars": 64,
                    "min_words": 8,
                    "min_quality": 0.4,
                    "script_sanity": True,
                },
                "release_use": "internal",
            }
        )
    )
    return recipe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequence-length", default=32, type=int)
    args = parser.parse_args()
    make_fixture(args.output_dir, sequence_length=args.sequence_length)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
