"""Describe how an outfit's palette colors relate, not just what they are called.

All measures are computed in CIELAB from the share-weighted outfit palette:

- lightness, saturation, temperature, and contrast from L*, C*, h, and CIEDE2000;
- the hue scheme (monochrome, analogous, complementary, ...) from hue angles of
  the chromatic colors;
- colour emotions from Ou, Luo, Woodcock & Wright (2004), "A study of colour
  emotion and colour preference. Part I", Color Res. Appl. 29(3): activity,
  weight, and heat;
- pairwise harmony from Ou & Luo (2006), "A colour harmony model for
  two-colour combinations", Color Res. Appl. 31(3);
- a mood placed on a warm-cool / soft-hard grid in the spirit of Kobayashi's
  Color Image Scale. The mood words are this project's own (config/color_moods.yaml).

Lighting shifts every sampled color together, so the relative measures
(contrast, harmony, scheme) are more trustworthy than absolute ones.
"""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from .color_names import ciede2000, hex_to_lab
from .schemas import ColorRelations, PaletteColor

MOODS = Path(__file__).resolve().parents[1] / "config" / "color_moods.yaml"
CHROMATIC = 15.0  # C* below this reads as a neutral (grey, black, white, beige-grey).


def _lch(lab: np.ndarray) -> tuple[float, float, float]:
    lightness, a, b = (float(value) for value in lab)
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def _cos(degrees: float) -> float:
    return math.cos(math.radians(degrees))


def colour_emotions(lab: np.ndarray) -> tuple[float, float, float]:
    """Ou et al. (2004) colour activity, weight, and heat for one color.

    The hue term of colour weight is scaled by chroma: a neutral has no
    meaningful hue angle, and sampled greys would otherwise get a random one.
    """
    lightness, a, b = (float(value) for value in lab)
    _, chroma, hue = _lch(lab)
    activity = -2.1 + .06 * math.sqrt((lightness - 50) ** 2 + (a - 3) ** 2 + ((b - 17) / 1.4) ** 2)
    weight = -1.8 + .04 * (100 - lightness) + .45 * _cos(hue - 100) * min(1.0, chroma / CHROMATIC)
    heat = -.5 + .02 * chroma ** 1.07 * _cos(hue - 50)
    return activity, weight, heat


def pair_harmony(first: np.ndarray, second: np.ndarray) -> float:
    """Ou & Luo (2006) harmony of two colors; above 0 tends to look harmonious, below 0 clashing."""
    l1, c1, h1 = _lch(first)
    l2, c2, h2 = _lch(second)
    delta_hue = 2 * math.sqrt(c1 * c2) * math.sin(math.radians(((h2 - h1 + 180) % 360 - 180) / 2))
    delta_chroma = math.sqrt(delta_hue ** 2 + ((c1 - c2) / 1.46) ** 2)
    h_c = .04 + .53 * math.tanh(.8 - .045 * delta_chroma)
    h_lsum = .28 + .54 * math.tanh(-3.88 + .029 * (l1 + l2))
    h_dl = .14 + .15 * math.tanh(-2 + .2 * abs(l1 - l2))

    def hue_effect(lightness: float, chroma: float, hue: float) -> float:
        e_c = .5 + .5 * math.tanh(-2 + .5 * chroma)
        h_s = -.08 - .14 * math.sin(math.radians(hue + 50)) - .07 * math.sin(math.radians(2 * hue + 90))
        x = (90 - hue) / 10
        e_y = (.22 * lightness - 12.8) / 10 * math.exp(x - math.exp(x))
        return e_c * (h_s + e_y)

    return h_c + h_lsum + h_dl + hue_effect(l1, c1, h1) + hue_effect(l2, c2, h2)


def _hue_gap(first: float, second: float) -> float:
    return abs((second - first + 180) % 360 - 180)


def _scheme(hues: list[float], has_neutrals: bool) -> str:
    if not hues:
        return "neutral"
    if len(hues) == 1:
        return "accented neutral" if has_neutrals else "monochrome"
    gaps = [_hue_gap(a, b) for a, b in combinations(hues, 2)]
    if max(gaps) <= 30:
        return "monochrome"
    if max(gaps) <= 60:
        return "analogous"
    if len(hues) >= 3 and min(gaps) >= 90:
        return "triadic"
    if any(150 <= gap <= 210 for gap in gaps):
        return "complementary"
    return "contrasting"


@lru_cache(maxsize=1)
def _mood_table() -> dict:
    with MOODS.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def color_relations(palette: list[PaletteColor]) -> ColorRelations | None:
    """Summarize the palette's light, chroma, temperature, contrast, scheme, harmony, and mood."""
    colors = [(color, hex_to_lab(color.hex)) for color in palette if color.share > 0]
    if not colors:
        return None
    total = sum(color.share for color, _ in colors)
    weights = [color.share / total for color, _ in colors]
    lch = [_lch(lab) for _, lab in colors]
    emotions = [colour_emotions(lab) for _, lab in colors]

    def mean(values) -> float:
        return float(sum(weight * value for weight, value in zip(weights, values)))

    lightness, chroma = mean(l for l, _, _ in lch), mean(c for _, c, _ in lch)
    activity, weight, heat = (mean(values) for values in zip(*emotions))
    # Hue warmth, strongest for vivid colors; neutrals pull the outfit toward the middle.
    warmth = mean(_cos(h - 50) * min(1.0, c / 40) if c >= CHROMATIC else 0.0 for _, c, h in lch)
    differences = [float(ciede2000(a, b[None, :])[0]) for (_, a), (_, b) in combinations(colors, 2)]
    contrast_value = max(differences, default=0.0)
    pairs = [(weights[i] * weights[j], pair_harmony(colors[i][1], colors[j][1]))
             for i, j in combinations(range(len(colors)), 2)]
    harmony_score = (sum(w * h for w, h in pairs) / sum(w for w, _ in pairs)) if pairs else None
    # Hard reads as heavy and high-contrast, soft as light and close-valued (Kobayashi's vertical axis).
    hardness = .5 * weight + .5 * (contrast_value - 25) / 25

    contrast = "low" if contrast_value < 15 else "medium" if contrast_value < 35 else "high"
    temperature = "warm" if warmth > .2 else "cool" if warmth < -.2 else "neutral"
    texture = "hard" if hardness > .8 else "soft" if hardness < -.3 else "medium"
    table = _mood_table()
    moods = list(table["grid"][texture][temperature])
    saturation = "muted" if chroma < 20 else "vivid" if chroma > 45 else "moderate"
    # Ou & Luo's scale centres on 0; the band keeps classic pairs such as navy and beige out of "clashing".
    harmony = ("harmonious" if harmony_score is not None and harmony_score >= .25 else
               "clashing" if harmony_score is not None and harmony_score < -.3 else
               "balanced" if harmony_score is not None else "single")
    for key in (saturation, harmony):
        moods += [word for word in table.get("modifiers", {}).get(key, []) if word not in moods]
    return ColorRelations(
        lightness="light" if lightness > 70 else "dark" if lightness < 40 else "mid",
        saturation=saturation, temperature=temperature, contrast=contrast,
        scheme=_scheme([h for _, c, h in lch if c >= CHROMATIC], any(c < CHROMATIC for _, c, _ in lch)),
        harmony=harmony, moods=moods,
        activity=round(activity, 3), weight=round(weight, 3), heat=round(heat, 3),
        harmony_score=None if harmony_score is None else round(harmony_score, 3),
        warmth=round(warmth, 3), hardness=round(hardness, 3), contrast_value=round(contrast_value, 2))
