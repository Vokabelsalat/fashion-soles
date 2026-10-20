from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pytest
from PIL import Image

from fashion_soles.fashion_detector import PieceProposal
from fashion_soles.video import sample_video, select_video_pieces


def make_video(frame_count=6, fps=2):
    with NamedTemporaryFile(suffix=".mp4") as file:
        writer = cv2.VideoWriter(file.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (24, 16))
        assert writer.isOpened()
        for index in range(frame_count):
            writer.write(np.full((16, 24, 3), (index * 30, 50, 60), dtype=np.uint8))
        writer.release()
        return Path(file.name).read_bytes()


def test_video_decodes_bounded_evenly_spaced_frames():
    frames = sample_video(make_video(), ".mp4", max_frames=3)
    assert len(frames) == 3
    assert [frame.seconds for frame in frames] == [0, 1, 2.5]
    assert all(frame.image.size == (24, 16) and frame.image.mode == "RGB" for frame in frames)
    assert frames[0].image.getpixel((0, 0)) != frames[-1].image.getpixel((0, 0))


def test_video_rejects_invalid_size_duration_and_content():
    with pytest.raises(ValueError, match="under 100 MB"):
        sample_video(b"", ".mp4")
    with pytest.raises(ValueError, match="could not be decoded"):
        sample_video(b"not a video", ".mp4")
    with pytest.raises(ValueError, match="no longer than 60 seconds"):
        sample_video(make_video(frame_count=62, fps=1), ".mp4")
    with pytest.raises(ValueError, match="Choose an MP4"):
        sample_video(b"data", ".avi")


def test_repeated_labels_keep_best_frame_without_losing_new_labels():
    mask = Image.new("L", (10, 10), 255)
    frames = [
        [PieceProposal("shirt", .7, (0, 0, 10, 10), mask, "intersection", ("#FFFFFF",))],
        [PieceProposal("shirt", .9, (0, 0, 10, 10), mask, "intersection", ("#FFFFFF",)),
         PieceProposal("shoe", .8, (0, 0, 10, 10), mask, "intersection", ("#FFFFFF",))],
    ]
    merged = select_video_pieces(frames)
    assert [(item.proposal.label, item.frame_index, item.seen_frames) for item in merged] == [
        ("shirt", 1, 2), ("shoe", 1, 1)]
