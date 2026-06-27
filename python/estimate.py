"""
Reference implementation of the heuristic estimator, mirroring go/estimator.go.

Used by calculate_discount.py to compute a GLOBAL correction factor:
    discount = real_tokens / estimate(table, text, discount=1.0)

Keeping this in lockstep with estimator.go is essential — any divergence in
char-class handling makes the calibrated discount wrong. The char classes,
ranges, and per-class weights here must match estimator.go exactly.
"""

from __future__ import annotations

import math
import unicodedata

CJK_START = 0x4E00
CJK_END   = 0x9FFF

ASCII_SPACE_TOKEN = 0.2
TAB_TOKEN = 0.8
NEWLINE_TOKEN = 0.5


def _is_latin(cp: int) -> bool:
    return (0x61 <= cp <= 0x7A) or (0x41 <= cp <= 0x5A)


def _is_nd_digit(ch: str) -> bool:
    # mirrors Go unicode.IsDigit (Unicode category Nd)
    return unicodedata.category(ch) == "Nd"


def classify_text(text: str) -> str:
    """Return 'zh', 'en', or 'mixed' based on CJK character ratio."""
    if not text:
        return "mixed"
    cjk = sum(1 for c in text if CJK_START <= ord(c) <= CJK_END)
    ratio = cjk / len(text)
    if ratio >= 0.6:
        return "zh"
    if ratio <= 0.1:
        return "en"
    return "mixed"


def estimate(table: bytes | None, text: str, discount: float | dict = 1.0) -> float:
    if isinstance(discount, dict):
        cat = classify_text(text)
        discount = discount.get(cat, discount.get("mixed", 1.0))
    runes = list(text)
    n = len(runes)
    tokens = 0.0

    i = 0
    while i < n:
        ch = runes[i]
        cp = ord(ch)

        # CJK Unified Ideographs (main block) — table lookup
        if CJK_START <= cp <= CJK_END:
            if table is not None:
                tokens += table[cp - CJK_START]
            else:
                tokens += 1.5
            i += 1

        # CJK Extension A / Compatibility Ideographs — fallback
        elif (0x3400 <= cp <= 0x4DBF) or (0xF900 <= cp <= 0xFAFF):
            tokens += 1.5
            i += 1

        # Latin letter run — scale with length
        elif _is_latin(cp):
            j = i + 1
            while j < n and _is_latin(ord(runes[j])):
                j += 1
            word_len = j - i
            tokens += math.ceil(word_len / 4.0)
            i = j

        # Hiragana / Katakana
        elif (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
            tokens += 1.0
            i += 1

        # Korean syllables
        elif 0xAC00 <= cp <= 0xD7AF:
            tokens += 1.5
            i += 1

        # Digit run (Nd)
        elif _is_nd_digit(ch):
            j = i + 1
            while j < n and _is_nd_digit(runes[j]):
                j += 1
            tokens += (j - i) * 0.5
            i = j

        # Newlines
        elif ch == "\n" or ch == "\r":
            tokens += NEWLINE_TOKEN
            i += 1

        # ASCII whitespace often merges into adjacent tokens, especially in
        # code, JSON, and Markdown indentation.
        elif ch == "\t":
            tokens += TAB_TOKEN
            i += 1
        elif ch == " ":
            tokens += ASCII_SPACE_TOKEN
            i += 1

        # CJK / fullwidth / general punctuation
        elif (0x2000 <= cp <= 0x206F) or (0x3000 <= cp <= 0x303F) or (0xFF00 <= cp <= 0xFFEF):
            tokens += 1.0
            i += 1

        # ASCII punctuation (printable, non-alphanumeric)
        elif 0x21 <= cp <= 0x7E and not ch.isalnum():
            tokens += 0.7
            i += 1

        # Everything else (emoji, rare symbols, …)
        else:
            tokens += 3.0
            i += 1

    return tokens * discount
