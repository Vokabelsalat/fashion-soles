from __future__ import annotations

import io
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
ALLOWED = {"JPEG", "PNG", "WEBP"}


def prepare_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD:
        raise ValueError("Choose a non-empty image under 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in ALLOWED:
                raise ValueError("Only JPG, PNG, and WEBP images are supported.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image has too many pixels (maximum 20 million).")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The image is corrupt or cannot be decoded.") from exc
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    return image.copy()  # New RGB image has no EXIF metadata.


def jpeg_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=88)
    return output.getvalue()
