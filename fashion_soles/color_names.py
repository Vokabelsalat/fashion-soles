"""Name sampled colors with the closest entry from a bundled name list.

Two modes: "basic" uses the CSS/X11 named colors (grey spellings, one name per
hex), "whimsical" uses the curated "best of" subset of meodai/color-names
(MIT, see data/colornames.LICENSE.txt). Matching uses CIEDE2000 in CIELAB, so
the chosen name is the one that looks closest, not the one nearest in RGB.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
COLOR_MODES = {"basic": DATA_DIR / "colornames.css.csv", "whimsical": DATA_DIR / "colornames.bestof.csv"}


def _hex_to_rgb(values: list[str]) -> np.ndarray:
    return np.array([[int(value[i:i + 2], 16) for i in (1, 3, 5)] for value in values], dtype=float) / 255


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0, 1] to CIELAB under D65."""
    linear = np.where(rgb > .04045, ((rgb + .055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = np.einsum("ij,jk->ik", linear, np.array([[.4124564, .2126729, .0193339],
                                                   [.3575761, .7151522, .1191920],
                                                   [.1804375, .0721750, .9503041]]))
    xyz /= np.array([.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _ciede2000(lab: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Distance from one Lab color to every row of ``table`` (Sharma et al. 2005)."""
    l1, a1, b1 = lab
    l2, a2, b2 = table.T
    c_mean = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = .5 * (1 - np.sqrt(c_mean ** 7 / (c_mean ** 7 + 25 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dl, dc = l2 - l1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    l_mean, cp_mean = (l1 + l2) / 2, (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_mean = np.where(c1p * c2p == 0, h_sum,
                      np.where(np.abs(h1p - h2p) <= 180, h_sum / 2,
                               np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)))
    t = (1 - .17 * np.cos(np.radians(h_mean - 30)) + .24 * np.cos(np.radians(2 * h_mean))
         + .32 * np.cos(np.radians(3 * h_mean + 6)) - .20 * np.cos(np.radians(4 * h_mean - 63)))
    sl = 1 + .015 * (l_mean - 50) ** 2 / np.sqrt(20 + (l_mean - 50) ** 2)
    sc = 1 + .045 * cp_mean
    sh = 1 + .015 * cp_mean * t
    rt = (-2 * np.sqrt(cp_mean ** 7 / (cp_mean ** 7 + 25 ** 7))
          * np.sin(np.radians(60 * np.exp(-((h_mean - 275) / 25) ** 2))))
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dhp / sh) ** 2 + rt * (dc / sc) * (dhp / sh))


@lru_cache(maxsize=len(COLOR_MODES))
def _table(mode: str) -> tuple[list[str], np.ndarray]:
    with COLOR_MODES[mode].open(encoding="utf-8", newline="") as handle:
        rows = [(row["name"], row["hex"]) for row in csv.DictReader(handle)
                if re.fullmatch(r"#[0-9a-fA-F]{6}", row["hex"])]
    names = [name for name, _ in rows]
    return names, _rgb_to_lab(_hex_to_rgb([value for _, value in rows]))


@lru_cache(maxsize=4096)
def fashion_color_name(value: str, mode: str = "basic") -> str:
    """Return the closest name, e.g. ``"Silver"`` (basic) or ``"Couch Potato"`` (whimsical) for ``#C2B4A7``."""
    if mode not in COLOR_MODES:
        raise ValueError(f"Unknown color name mode: {mode}")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return "unknown"
    names, table = _table(mode)
    lab = _rgb_to_lab(_hex_to_rgb([value]))[0]
    return names[int(np.argmin(_ciede2000(lab, table)))]
