"""Describe a garment's surface and pattern from the pixels inside its mask, without a model.

Several cheap measures, each aimed at one pattern trait:

- density of crisp edges (Canny with high thresholds) inside the slightly eroded
  mask: plain / textured / busy. Folds and shading are soft and mostly drop out;
- orientation histogram of strong gradients: one dominant direction (stripes),
  two at about 90 degrees (checks), or none (dots, flowers);
- the strongest peak of the Fourier spectrum against the average: regular
  repeats (stripes, checks, dots) versus irregular prints and folds;
- round, similar-sized blobs in the minority tone: polka dots;
- distinct colour clusters and the share of edges on straight lines: organic
  ("floral") versus geometric prints;
- how concentrated the edges are: a single graphic (logo, motif) versus an
  all-over print.

"Floral print" means an organic, curvy, multi-coloured print; paisley or
animal prints land there too. Folds and shadows also create edges, so the
thresholds favour "plain" and "textured" when the evidence is weak: sparse or
faint prints (a small faded motif) read as plain.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

SIDE = 256  # Analysis resolution: enough for stripes and dots, fast on every frame.


@dataclass(frozen=True)
class Texture:
    surface: str  # plain, textured, or busy
    pattern: str  # none, striped, checked, dotted, floral print, geometric print, graphic print
    edge_density: float
    orientation: float  # Share of gradient energy in the dominant direction (0-1).
    cross: float  # Share in a second direction about 90 degrees away (0-1).
    regularity: float  # Strongest spectral peak over the mean spectrum, log10.
    colors: int
    straightness: float  # Share of edge pixels on straight segments (0-1).
    concentration: float  # Share of edges in the densest eighth of the area (0-1).
    dots: int

    @property
    def label(self) -> str:
        return self.pattern if self.pattern != "none" else self.surface


def _orientation(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    values = magnitude[mask]
    strong = mask & (magnitude > max(20.0, float(np.percentile(values, 80)) if values.size else 0.0))
    if strong.sum() < 50:
        return 0.0, 0.0
    angles = np.degrees(np.arctan2(gy[strong], gx[strong])) % 180
    histogram, _ = np.histogram(angles, bins=18, range=(0, 180), weights=magnitude[strong])
    histogram = histogram / histogram.sum()
    smooth = (np.roll(histogram, 1) + histogram + np.roll(histogram, -1))  # 30-degree windows.
    first = int(np.argmax(smooth))
    away = np.abs((np.arange(18) - first + 9) % 18 - 9)  # Bin distance on the half circle.
    second = int(np.argmax(np.where(away >= 7, smooth, 0)))  # At least 70 degrees away.
    return float(smooth[first]), float(smooth[second]) if away[second] >= 7 else 0.0


def _regularity(gray: np.ndarray, mask: np.ndarray) -> float:
    patch = np.where(mask, gray - gray[mask].mean(), 0).astype(np.float32)
    patch *= np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1])).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2(patch))) ** 2
    rows, cols = np.indices(power.shape)
    radius = np.hypot(rows - power.shape[0] / 2, cols - power.shape[1] / 2)
    band = power[(radius >= 6) & (radius <= min(power.shape) / 2)]  # Skip folds and shading.
    if not band.size or band.max() <= 0:  # A perfectly uniform region has no spectrum at all.
        return 0.0
    return float(np.log10(band.max() / max(band.mean(), 1e-9)))


def _colors(lab: np.ndarray, mask: np.ndarray) -> int:
    pixels = lab[mask].astype(np.float32)
    if len(pixels) < 50:
        return 1
    pixels = pixels[np.linspace(0, len(pixels) - 1, min(len(pixels), 4000), dtype=int)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=4) / len(labels)
    kept: list[np.ndarray] = []
    for center in centers[np.argsort(shares)[::-1]][shares[np.argsort(shares)[::-1]] >= .08]:
        if all(np.linalg.norm(center - other) > 25 for other in kept):  # OpenCV Lab units, about ΔE 20.
            kept.append(center)
    return max(1, len(kept))


def _dots(gray: np.ndarray, mask: np.ndarray) -> int:
    values = gray[mask]
    threshold, _ = cv2.threshold(values.reshape(1, -1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (gray > threshold) & mask
    minority = bright if bright.sum() < mask.sum() / 2 else (~bright & mask)
    count, _, stats, _ = cv2.connectedComponentsWithStats(minority.astype(np.uint8), connectivity=8)
    area = mask.sum()
    sizes = []
    for index in range(1, count):
        x, y, w, h, pixels = stats[index]
        if not area * .0004 <= pixels <= area * .02 or not .5 <= w / max(h, 1) <= 2:
            continue
        # A disc fills its enclosing circle (about 0.85-1.0); a square overfills it (1.27),
        # triangles and half-squares underfill it (about 0.64).
        if .75 <= pixels / (np.pi * (max(w, h) / 2) ** 2) <= 1.05:
            sizes.append(pixels)
    if len(sizes) < 6:
        return 0
    sizes = np.array(sizes, dtype=float)
    return len(sizes) if sizes.std() / sizes.mean() < .6 else 0


def _straightness(edges: np.ndarray) -> float:
    total = int(np.count_nonzero(edges))
    if total < 50:
        return 0.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=SIDE // 10, maxLineGap=3)
    length = 0.0 if lines is None else float(sum(np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in lines.reshape(-1, 4)))
    return min(1.0, length / total)


def _concentration(edges: np.ndarray, mask: np.ndarray) -> float:
    rows, cols = np.array_split(np.arange(edges.shape[0]), 8), np.array_split(np.arange(edges.shape[1]), 8)
    cells = [(int(edges[np.ix_(r, c)].astype(bool).sum()), int(mask[np.ix_(r, c)].sum())) for r in rows for c in cols]
    inside = sorted((count for count, area in cells if area > 0), reverse=True)
    if not inside or not sum(inside):
        return 0.0
    return float(sum(inside[:max(1, len(inside) // 8)]) / sum(inside))


def analyse_texture(image: Image.Image, mask: Image.Image | None) -> Texture | None:
    """Return surface and pattern for the masked region, or None when it is too small to judge."""
    if mask is None:
        return None
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
    bounds = mask.convert("L").getbbox()
    if bounds is None:
        return None
    crop = image.convert("RGB").crop(bounds)
    region = mask.convert("L").crop(bounds)
    if np.count_nonzero(np.asarray(region)) < 2500:  # Too few real pixels; upscaling would invent detail.
        return None
    scale = SIDE / max(crop.size)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    rgb = np.asarray(crop.resize(size, Image.Resampling.LANCZOS))
    inside = np.asarray(region.resize(size, Image.Resampling.NEAREST)) > 0
    # Erode so the garment outline and the background seam do not count as fabric edges.
    radius = max(2, round(.03 * SIDE))
    inside = cv2.erode(inside.astype(np.uint8), np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)) > 0
    if inside.sum() < 1500:
        return None
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.GaussianBlur(lab[:, :, 0], (3, 3), 0)
    # High thresholds keep crisp printed edges; folds and shading in real photos are soft and mostly drop out.
    edges = cv2.Canny(gray, 80, 180)
    edges[~inside] = 0
    density = float(np.count_nonzero(edges) / inside.sum())
    orientation, cross = _orientation(gray.astype(np.float32), inside)
    regularity = _regularity(gray.astype(np.float32), inside)
    colors = _colors(lab, inside)
    dots = _dots(gray, inside)
    straightness = _straightness(edges)
    concentration = _concentration(edges, inside)

    surface = "plain" if density < .04 else "busy" if density > .12 else "textured"
    pattern = "none"
    regular = regularity >= 2.8  # Plain, folds, and irregular prints stay near 1.5-2.3; repeats reach 3+.
    # No pattern on plain surfaces: in real photos a hand or a crease also leaves a few crisp, clustered
    # edges, so a small graphic on plain fabric cannot be told apart reliably without a model.
    if surface == "plain":
        pass
    elif dots and straightness < .3:
        pattern = "dotted"
    elif regular and orientation >= .25 and cross >= .2:
        pattern = "checked"
    elif regular and orientation >= .45 and cross < .15:
        pattern = "striped"
    elif concentration >= .45 and density < .2:
        pattern = "graphic print"
    elif density >= .03 and colors >= 2:
        pattern = "geometric print" if straightness >= .1 else "floral print"
    return Texture(surface, pattern, round(density, 4), round(orientation, 3), round(cross, 3),
                   round(regularity, 3), colors, round(straightness, 3), round(concentration, 3), dots)
