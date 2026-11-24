"""Conservative Unicode-script sanity check, NOT language identification.

Tamil script is strongly associated with Tamil; Devanagari is shared by Hindi
and other languages. English is usually written in Latin script. Code-mixed
texts can be rejected at strict thresholds. Real language ID needs a separately
licensed/validated model (e.g. fastText lid.176) and human evaluation.
"""
from __future__ import annotations

import unicodedata


def script_fractions(text: str) -> dict[str, float]:
    counts = {"latin": 0, "tamil": 0, "devanagari": 0, "other": 0}
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        code = ord(ch)
        if (0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A or
                0x00C0 <= code <= 0x024F):
            counts["latin"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            counts["tamil"] += 1
        elif 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        else:
            counts["other"] += 1
    total = sum(counts.values())
    return {key: round(val / total, 5) if total else 0.0
            for key, val in counts.items()}


def script_consistent(text: str, language: str, *, min_fraction: float = 0.65
                      ) -> tuple[bool, dict[str, float]]:
    if not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_fraction must be within (0,1]")
    expected = {"en": "latin", "ta": "tamil", "hi": "devanagari"}.get(language)
    if expected is None:
        raise ValueError("unsupported language for script sanity")
    shares = script_fractions(text)
    return shares[expected] >= min_fraction, shares
