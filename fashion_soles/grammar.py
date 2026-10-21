"""A small Tracery-style grammar engine for the poems.

Rules are lists of alternatives. ``#symbol#`` expands another rule (or a context
value), and modifiers follow dots: ``#item.a#`` adds "a"/"an", ``#color.cap#``
capitalizes the first letter, ``#lead.title#`` every word. A rule may be a dict
instead of a list; it is then narrowed by the expansion's keys in order (for
example tone, narrative function, palette size) until a list remains, with "any"
as the fallback key. So one symbol can carry
lines for every tone and every narrative function, and nest freely.

Choices come from a seeded random generator, so the same seed always yields the
same text. Within one expansion run an alternative is not reused until all of a
symbol's alternatives have been used.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

TOKEN = re.compile(r"#([^#\s]+)#")


def _article(text: str) -> str:
    return f"{'an' if text[:1].lower() in 'aeiou' else 'a'} {text}"


MODIFIERS = {"cap": lambda text: text[:1].upper() + text[1:], "a": _article,
             "title": lambda text: " ".join(word[:1].upper() + word[1:] for word in text.split(" "))}

_cache: dict[Path, tuple[float, dict]] = {}


def load_grammar(path: Path) -> dict:
    """Read the grammar file, again whenever it changes on disk (so edits apply on the next rerun)."""
    modified = path.stat().st_mtime
    cached = _cache.get(path)
    if cached is None or cached[0] != modified:
        with path.open(encoding="utf-8") as handle:
            cached = (modified, yaml.safe_load(handle))
        _cache[path] = cached
    return cached[1]


class Grammar:
    def __init__(self, rules: dict, seed: int, context: dict[str, str] | None = None):
        self.rules = rules
        self.rng = random.Random(seed)
        self.context = dict(context or {})
        self.used: set[tuple[str, str]] = set()

    def has(self, symbol: str, keys: list[str]) -> bool:
        try:
            self._options(symbol, keys)
        except KeyError:
            return False
        return True

    def _options(self, symbol: str, keys: list[str]) -> list[str]:
        node = self.rules[symbol]
        while isinstance(node, dict):
            key = next((key for key in [*keys, "any"] if key in node), None)
            if key is None:
                raise KeyError(f"No alternatives for '{symbol}' under keys {keys}")
            node = node[key]
        return node

    def expand(self, symbol: str, keys: list[str], extra: dict[str, str] | None = None, depth: int = 0) -> str:
        """Pick one alternative of ``symbol`` and expand its tokens."""
        if depth > 12:
            raise RecursionError(f"Grammar recursion too deep at '{symbol}'")
        options = self._options(symbol, keys)
        fresh = [option for option in options if (symbol, option) not in self.used] or options
        choice = self.rng.choice(fresh)
        self.used.add((symbol, choice))
        return self.fill(choice, keys, extra, depth)

    def fill(self, text: str, keys: list[str], extra: dict[str, str] | None = None, depth: int = 0) -> str:
        context = {**self.context, **(extra or {})}

        def replace(match: re.Match) -> str:
            name, *modifiers = match.group(1).split(".")
            value = context[name] if name in context else self.expand(name, keys, extra, depth + 1)
            for modifier in modifiers:
                value = MODIFIERS[modifier](value)
            return value

        return TOKEN.sub(replace, text)
