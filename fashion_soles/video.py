"""Bounded, local video sampling and cross-frame piece selection."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import numpy as np
from PIL import Image

from .fashion_detector import Detection, PieceProposal, intersect_detection, preview_selected_pieces
from .segmentation import segment_clothes

MAX_VIDEO_SECONDS = 60
MAX_VIDEO_PIXELS = 20_000_000
MAX_SAMPLED_FRAMES = 5
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm"}
OVERLAY_FPS = 6
OVERLAY_MAX_SIDE = 960


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


def _boxes_at_time(frames: list[VideoFrame], proposals: list[list[PieceProposal]],
                   labels: set[str], seconds: float, size: tuple[int, int]) -> list[Detection]:
    """Interpolate boxes only between neighboring samples that both saw a label."""
    if len(frames) != len(proposals):
        raise ValueError("Video frames and detections must correspond.")
    result = []
    for label in sorted(labels):
        seen = []
        for index, (frame, pieces) in enumerate(zip(frames, proposals)):
            matches = [piece for piece in pieces if piece.label == label]
            if matches:
                piece = max(matches, key=lambda item: item.score)
                normalized = tuple(value / frame.image.size[axis % 2]
                                   for axis, value in enumerate(piece.box))
                seen.append((index, frame.seconds, piece.score, normalized))
        if not seen:
            continue
        before = max((item for item in seen if item[1] <= seconds), default=None, key=lambda item: item[1])
        after = min((item for item in seen if item[1] >= seconds), default=None, key=lambda item: item[1])
        if before and after and before[0] != after[0] and after[0] == before[0] + 1:
            fraction = (seconds - before[1]) / max(after[1] - before[1], 1e-6)
            box = tuple(a + (b - a) * fraction for a, b in zip(before[3], after[3]))
            score = min(before[2], after[2])
        else:
            nearest = min(seen, key=lambda item: abs(item[1] - seconds))
            # A single observation should not follow the wearer across the whole clip.
            sample_spacing = (frames[-1].seconds - frames[0].seconds) / max(1, len(frames) - 1)
            if abs(nearest[1] - seconds) > max(.25, sample_spacing / 2):
                continue
            box, score = nearest[3], nearest[2]
        result.append(Detection(label, score, tuple(round(value * size[index % 2])
                                                     for index, value in enumerate(box))))
    return result


def render_annotated_video(data: bytes | BinaryIO, suffix: str, frames: list[VideoFrame],
                           frame_proposals: list[list[PieceProposal]], labels: set[str]) -> bytes:
    """Render a silent H.264 replay with sampled boxes and fresh coarse masks.

    This is an approximation between detector samples, not object tracking.
    """
    if suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("Choose an MP4, MOV, or WEBM video.")
    if not frames or len(frames) != len(frame_proposals):
        raise ValueError("Video frames and detections must correspond.")
    import cv2

    with NamedTemporaryFile(suffix=suffix.lower()) as source_file, NamedTemporaryFile(suffix=".mp4") as output_file:
        source = BytesIO(data) if isinstance(data, bytes) else data
        source.seek(0)
        copyfileobj(source, source_file, length=1024 * 1024)
        source.seek(0)
        source_file.flush()
        capture = cv2.VideoCapture(source_file.name)
        writer = None
        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not capture.isOpened() or not np.isfinite(source_fps) or source_fps <= 0 or count <= 0:
                raise ValueError("The video could not be decoded.")
            if width <= 0 or height <= 0 or width * height > MAX_VIDEO_PIXELS or count / source_fps > MAX_VIDEO_SECONDS:
                raise ValueError("Video exceeds the supported size or duration.")
            scale = min(1., OVERLAY_MAX_SIDE / max(width, height))
            size = (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))
            fps = min(source_fps, OVERLAY_FPS)
            writer = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*"avc1"), fps, size)
            if not writer.isOpened():
                raise RuntimeError("H.264 video encoding is unavailable in this OpenCV installation.")
            total = max(1, int(np.ceil(count / source_fps * fps)))
            for output_index in range(total):
                seconds = output_index / fps
                capture.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, round(seconds * source_fps)))
                success, bgr = capture.read()
                if not success or bgr is None:
                    break
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                boxes = _boxes_at_time(frames, frame_proposals, labels, seconds, size)
                if boxes:
                    seg = segment_clothes(image)
                    pieces = [intersect_detection(box, seg.mask, seg.accessory_mask, size) for box in boxes]
                    image = preview_selected_pieces(image, pieces)
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        finally:
            capture.release()
            if writer is not None:
                writer.release()
        output_file.seek(0)
        result = output_file.read()
        if not result:
            raise RuntimeError("No annotated video frames could be encoded.")
        return result


def key_frame_index(segmentations: list) -> int:
    """Pick the sampled frame whose MediaPipe clothes and accessory masks cover the most pixels."""
    if not segmentations:
        raise ValueError("At least one segmentation is required.")
    areas = []
    for seg in segmentations:
        masks = [np.asarray(mask.convert("L")) > 0 for mask in (seg.mask, seg.accessory_mask) if mask is not None]
        areas.append(int(np.count_nonzero(np.logical_or.reduce(masks))) if masks else 0)
    return max(range(len(areas)), key=lambda index: areas[index]) if any(areas) else len(areas) // 2
