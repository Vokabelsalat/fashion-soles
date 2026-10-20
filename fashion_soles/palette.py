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


def color_histogram(image: Image.Image, mask: Image.Image | None = None, *, levels: int = 6,
                    top: int = 24, max_side: int = 480) -> list[tuple[str, float]]:
    """Bin pixels on a fixed RGB grid and return the most common bins as (mean hex, share).

    The grid is fixed (``levels`` steps per channel) so the whole-image and
    clothing-only histograms are directly comparable. Each bin is shown with the
    mean color of its pixels, not the grid cell's center.
    """
    if levels < 2 or top < 1:
        raise ValueError("Histogram needs at least two levels and one bin")
    scale = min(1.0, max_side / max(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    pixels = np.asarray(image.convert("RGB").resize(size, Image.Resampling.BILINEAR)).reshape(-1, 3)
    if mask is not None:
        pixels = pixels[np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)).reshape(-1) > 0]
    if not len(pixels):
        return []
    cells = (pixels.astype(np.int64) * levels // 256)
    keys = cells[:, 0] * levels * levels + cells[:, 1] * levels + cells[:, 2]
    counts = np.bincount(keys, minlength=levels ** 3)
    sums = np.stack([np.bincount(keys, weights=pixels[:, channel], minlength=levels ** 3)
                     for channel in range(3)], axis=1)
    order = [key for key in np.argsort(counts)[::-1][:top] if counts[key]]
    return [("#{:02X}{:02X}{:02X}".format(*np.rint(sums[key] / counts[key]).astype(int)),
             float(counts[key] / len(pixels))) for key in order]
