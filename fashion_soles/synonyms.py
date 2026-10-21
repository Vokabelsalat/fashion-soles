"""Synonyms from WordNet, so a motif repeated in one poem need not repeat its word.

Only the word's most common noun sense is used, and only single-word lemmas that
WordNet's usage counts show as reasonably common: other senses produce swaps such
as "duty" -> "tariff". Even then some fits are poor ("earth" -> "globe"); the
grammar file's ``synonyms`` section blocks those and adds hand-picked ones.

The WordNet data lives in models/nltk_data; download it with
``python scripts/download_language_data.py``. Without it, motifs simply repeat.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

NLTK_DATA = Path(__file__).resolve().parents[1] / "models" / "nltk_data"


@lru_cache(maxsize=1)
def _wordnet():
    try:
        import nltk
        if str(NLTK_DATA) not in nltk.data.path:
            nltk.data.path.insert(0, str(NLTK_DATA))
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        return wordnet
    except (ImportError, LookupError):
        return None


@lru_cache(maxsize=512)
def wordnet_synonyms(word: str) -> tuple[str, ...]:
    """Common single-word synonyms from the word's first noun sense; empty without WordNet."""
    wordnet = _wordnet()
    if wordnet is None or " " in word:
        return ()
    found = []
    for synset in wordnet.synsets(word, pos=wordnet.NOUN)[:1]:
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name != word.lower() and "_" not in name and lemma.count() >= 2 and name not in found:
                found.append(name)
    return tuple(found)


def alternatives(word: str, blocked: dict[str, list[str]], extra: dict[str, list[str]]) -> list[str]:
    """The word first, then its usable synonyms: hand-picked ones, then WordNet's minus the blocked."""
    options = [word]
    for synonym in [*extra.get(word, []), *wordnet_synonyms(word)]:
        if synonym not in options and synonym not in blocked.get(word, []):
            options.append(synonym)
    return options
