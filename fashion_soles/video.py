"""Bounded, local video sampling and cross-frame piece selection."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import numpy as np
from PIL import Image

from .fashion_detector import PieceProposal

MAX_VIDEO_SECONDS = 60
MAX_VIDEO_PIXELS = 20_000_000
MAX_SAMPLED_FRAMES = 5
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}


@dataclass(frozen=True)
class VideoFrame:
    image: Image.Image
    seconds: float


def sample_video(data: bytes | BinaryIO, suffix: str, *, max_frames: int = MAX_SAMPLED_FRAMES) -> list[VideoFrame]:
    """Decode a few evenly spaced frames without keeping the uploaded video."""
    if suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Choose an MP4, MOV, or WEBM video.")
    if not 1 <= max_frames <= MAX_SAMPLED_FRAMES:
        raise ValueError("Invalid frame sample count.")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video decoding needs OpenCV. Install the video requirements first.") from exc

    with NamedTemporaryFile(suffix=suffix.lower()) as temporary:
        source = BytesIO(data) if isinstance(data, bytes) else data
        source.seek(0)
        copyfileobj(source, temporary, length=1024 * 1024)
        if temporary.tell() == 0:
            raise ValueError("Choose a non-empty video.")
        temporary.flush()
        capture = cv2.VideoCapture(temporary.name)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
                raise ValueError("The video could not be decoded.")
            if width <= 0 or height <= 0 or width * height > MAX_VIDEO_PIXELS:
                raise ValueError("Video frames exceed the 20 million pixel limit.")
            duration = frame_count / fps
            if duration > MAX_VIDEO_SECONDS:
                raise ValueError("Choose a video no longer than 60 seconds.")
            indexes = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
            frames = []
            for index in dict.fromkeys(indexes.tolist()):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                success, bgr = capture.read()
                if not success or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                frames.append(VideoFrame(image.copy(), round(index / fps, 2)))
            if not frames:
                raise ValueError("No frames could be decoded from this video.")
            return frames
        finally:
            capture.release()


@dataclass(frozen=True)
class VideoPiece:
    proposal: PieceProposal
    frame_index: int
    seen_frames: int


def select_video_pieces(frame_proposals: list[list[PieceProposal]]) -> list[VideoPiece]:
    """Keep the clearest observation per label across sampled frames.

    This is a compact outfit summary, not object tracking: same-label pieces
    in different frames are treated as one item.
    """
    by_label: dict[str, VideoPiece] = {}
    seen: dict[str, set[int]] = {}
    for frame_index, proposals in enumerate(frame_proposals):
        for piece in proposals:
            if piece.mask is None or piece.mask.getbbox() is None:
                continue
            label = piece.label.strip().lower()
            seen.setdefault(label, set()).add(frame_index)
            candidate = VideoPiece(piece, frame_index, 0)
            current = by_label.get(label)
            if current is None or (piece.score, np.count_nonzero(np.asarray(piece.mask))) > (
                    current.proposal.score, np.count_nonzero(np.asarray(current.proposal.mask))):
                by_label[label] = candidate
    return [VideoPiece(item.proposal, item.frame_index, len(seen[label]))
            for label, item in by_label.items()]
