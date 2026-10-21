"""Syllables and rhymes from the CMU Pronouncing Dictionary, read locally.

The dictionary (BSD license, see models/cmudict/LICENSE) maps English words to
ARPAbet phones such as ``light  L AY1 T``; vowels carry a stress digit, so a
word's syllable count is its number of digits. Two words rhyme when everything
from their last stressed vowel onward matches. Words missing from the
dictionary (rare whimsical color names) fall back to counting vowel groups.

Download the data with ``python scripts/download_language_data.py``.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

DICTIONARY = Path(__file__).resolve().parents[1] / "models" / "cmudict" / "cmudict.dict"
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def available() -> bool:
    return DICTIONARY.exists()


@lru_cache(maxsize=1)
def _pronunciations() -> dict[str, list[list[str]]]:
    if not available():
        raise RuntimeError("The CMU dictionary is missing; run python scripts/download_language_data.py.")
    entries: dict[str, list[list[str]]] = {}
    with DICTIONARY.open(encoding="utf-8") as handle:
        for line in handle:
            word, _, phones = line.partition(" ")
            phones = phones.split("#", 1)[0].split()  # Some entries carry a trailing comment.
            entries.setdefault(re.sub(r"\(\d+\)$", "", word), []).append(phones)
    return entries


def _guess_syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups) - (1 if word.endswith("e") and not word.endswith(("le", "ee")) and len(groups) > 1 else 0)
    return max(1, count)


def syllables(word: str) -> int:
    phones = _pronunciations().get(word.lower())
    if phones:
        return sum(phone[-1].isdigit() for phone in phones[0])
    return _guess_syllables(word.lower())


def line_syllables(text: str) -> int:
    return sum(syllables(word) for word in WORD.findall(text.lower()))


def last_word(text: str) -> str:
    words = WORD.findall(text.lower())
    return words[-1] if words else ""


def rhyming_parts(word: str) -> set[str]:
    """Phones from the last stressed vowel to the end, for every pronunciation of the word."""
    parts = set()
    for phones in _pronunciations().get(word.lower(), []):
        stressed = [index for index, phone in enumerate(phones) if phone[-1] in "12"]
        if stressed:
            parts.add(" ".join(phone.rstrip("012") for phone in phones[stressed[-1]:]))
    return parts


def rhymes(first: str, second: str) -> bool:
    """True for two different words that share a rhyming part ("light" and "night", not "light" twice)."""
    return first.lower() != second.lower() and bool(rhyming_parts(first) & rhyming_parts(second))
