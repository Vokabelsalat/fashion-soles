from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from PIL import Image


@dataclass
class Segmentation:
    status: str
    reason: str = ""
    mask: Image.Image | None = None
    accessory_mask: Image.Image | None = None
    clothes_percentage: float | None = None


@lru_cache(maxsize=2)
def _segmenter(model_path: str):
    import mediapipe as mp
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True, output_confidence_masks=False)
    return mp, mp.tasks.vision.ImageSegmenter.create_from_options(options), Lock()


def segment_clothes(image: Image.Image, model_path: str | None = None) -> Segmentation:
    path = Path(model_path or os.getenv("SEGMENTATION_MODEL_PATH", "models/selfie_multiclass_256x256.tflite"))
    if not path.is_file():
        return Segmentation("Segmentation unavailable", f"Model asset missing: {path}")
    try:
        import numpy as np
        mp, segmenter, lock = _segmenter(str(path.resolve()))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
        with lock:
            result = segmenter.segment(mp_image)
        if result.category_mask is None:
            raise RuntimeError("No category mask returned")
        categories = np.squeeze(result.category_mask.numpy_view())
        if categories.ndim != 2:
            raise RuntimeError("Unexpected category mask shape")
        clothes = (categories == 4).astype("uint8") * 255
        accessories = (categories == 5).astype("uint8") * 255
        mask = Image.fromarray(clothes).resize(image.size, Image.Resampling.NEAREST)
        accessory_mask = Image.fromarray(accessories).resize(image.size, Image.Resampling.NEAREST)
        percentage = round(float(100 * np.mean(categories == 4)), 1)
        return Segmentation("Ready", mask=mask, accessory_mask=accessory_mask, clothes_percentage=percentage)
    except Exception as exc:
        return Segmentation("Segmentation unavailable", f"Local model could not run: {type(exc).__name__}: {exc}")


def overlay(image: Image.Image, mask: Image.Image, opacity: float = .45) -> Image.Image:
    opacity = max(0, min(1, opacity))
    tinted = Image.new("RGB", image.size, (176, 141, 87))
    alpha = mask.point(lambda value: int(value * opacity))
    return Image.composite(tinted, image, alpha)
