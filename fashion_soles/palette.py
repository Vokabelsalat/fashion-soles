"""Build an outfit palette from pixels inside selected garment masks."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np
from PIL import Image

from .schemas import PaletteColor

if TYPE_CHECKING:
    from .from_pieces import PieceInput


def _sample_pixels(items: Sequence[PieceInput], limit: int = 30000) -> np.ndarray:
    regions: list[np.ndarray] = []
    for item in items:
        mask = item.proposal.mask
        image = item.image
        if mask is not None and image is not None:
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.Resampling.NEAREST)
            pixels = np.asarray(image.convert("RGB"))[np.asarray(mask.convert("L")) > 0]
            if len(pixels):
                regions.append(pixels)
                continue
        # A stored proposal can still supply an approximate palette without its source image.
        if item.proposal.colors:
            value = item.proposal.colors[0]
            if len(value) == 7 and value.startswith("#"):
                try:
                    rgb = [int(value[index:index + 2], 16) for index in (1, 3, 5)]
                except ValueError:
                    continue
                area = int(np.count_nonzero(np.asarray(mask))) if mask is not None else 1
                regions.append(np.tile(np.array(rgb, dtype=np.uint8), (min(limit, max(1, area)), 1)))
    if not regions:
        return np.empty((0, 3), dtype=np.uint8)
    total = sum(len(region) for region in regions)
    samples = []
    for region in regions:
        count = min(len(region), max(1, round(limit * len(region) / total)))
        samples.append(region[np.linspace(0, len(region) - 1, count, dtype=int)])
    return np.concatenate(samples)


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(float)


def outfit_palette(items: Sequence[PieceInput]) -> list[PaletteColor]:
    """Return primary, secondary, then a substantial distant accent when present."""
    pixels = _sample_pixels(items)
    if len(pixels) == 0:
        return []
    strip = Image.fromarray(pixels.reshape(1, -1, 3), "RGB")
    quantized = strip.quantize(colors=12, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    entries = sorted(quantized.getcolors() or [], reverse=True)
    clusters: list[tuple[int, np.ndarray]] = []
    for count, index in entries:
        rgb = np.array(palette[index * 3:index * 3 + 3], dtype=float)
        rgb_lab = _lab(rgb.astype(np.uint8))[0]
        match = next((position for position, (_, color) in enumerate(clusters)
                      if np.linalg.norm(_lab(color.astype(np.uint8))[0] - rgb_lab) < 20), None)
        if match is None:
            clusters.append((count, rgb))
        else:
            previous_count, previous_rgb = clusters[match]
            clusters[match] = (previous_count + count,
                               (previous_rgb * previous_count + rgb * count) / (previous_count + count))
    clusters.sort(key=lambda cluster: -cluster[0])
    primary = clusters[0]
    secondary = next((cluster for cluster in clusters[1:]
                      if np.linalg.norm(_lab(cluster[1].astype(np.uint8))[0] -
                                        _lab(primary[1].astype(np.uint8))[0]) >= 22), None)
    chosen = [("primary", primary)]
    if secondary is not None:
        chosen.append(("secondary", secondary))
        accent = next((cluster for cluster in clusters[1:] if cluster is not secondary
                       and cluster[0] / len(pixels) >= .015
                       and all(np.linalg.norm(_lab(cluster[1].astype(np.uint8))[0] -
                                              _lab(existing[1].astype(np.uint8))[0]) >= 38
                               for _, existing in chosen)), None)
        if accent is not None:
            chosen.append(("accent", accent))
    return [PaletteColor(role=role,
                         hex="#{:02X}{:02X}{:02X}".format(*np.rint(rgb).astype(int)),
                         share=count / len(pixels))
            for role, (count, rgb) in chosen]
