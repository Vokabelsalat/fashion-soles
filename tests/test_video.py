from pathlib import Path
from io import BytesIO
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import pytest
from PIL import Image

from fashion_soles.fashion_detector import PieceProposal
from fashion_soles.segmentation import Segmentation
from fashion_soles.video import VideoFrame, _boxes_at_time, render_annotated_video, sample_video, select_video_pieces


def make_video(frame_count=6, fps=2):
    with NamedTemporaryFile(suffix=".mp4") as file:
        writer = cv2.VideoWriter(file.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (24, 16))
        assert writer.isOpened()
        for index in range(frame_count):
            writer.write(np.full((16, 24, 3), (index * 30, 50, 60), dtype=np.uint8))
        writer.release()
        return Path(file.name).read_bytes()


def test_video_decodes_bounded_evenly_spaced_frames():
    frames = sample_video(BytesIO(make_video()), ".mp4", max_frames=3)
    assert len(frames) == 3
    assert [frame.seconds for frame in frames] == [0, 1, 2.5]
    assert all(frame.image.size == (24, 16) and frame.image.mode == "RGB" for frame in frames)
    assert frames[0].image.getpixel((0, 0)) != frames[-1].image.getpixel((0, 0))


def test_video_rejects_empty_long_and_invalid_content():
    with pytest.raises(ValueError, match="non-empty video"):
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


def test_overlay_boxes_move_between_samples_and_respect_selection():
    image = Image.new("RGB", (24, 16), "white")
    mask = Image.new("L", image.size, 255)
    frames = [VideoFrame(image, 0), VideoFrame(image, 2)]
    proposals = [
        [PieceProposal("shirt", .9, (1, 1, 7, 9), mask, "intersection")],
        [PieceProposal("shirt", .9, (13, 1, 19, 9), mask, "intersection")],
    ]
    assert _boxes_at_time(frames, proposals, {"shirt"}, 1, image.size)[0].box == (7, 1, 13, 9)
    assert _boxes_at_time(frames, proposals, {"shoe"}, 1, image.size) == []


def test_render_annotated_video_encodes_masks_over_playing_frames(monkeypatch):
    source = make_video(frame_count=6, fps=2)
    frames = sample_video(source, ".mp4", max_frames=2)
    mask = Image.new("L", frames[0].image.size, 255)
    proposals = [[PieceProposal("shirt", .9, (2, 2, 12, 12), mask, "intersection")]
                 for _ in frames]
    monkeypatch.setattr("fashion_soles.video.segment_clothes",
                        lambda image: Segmentation("Ready", mask=Image.new("L", image.size, 255)))
    encoded = render_annotated_video(source, ".mp4", frames, proposals, {"shirt"})
    with NamedTemporaryFile(suffix=".mp4") as file:
        file.write(encoded)
        file.flush()
        capture = cv2.VideoCapture(file.name)
        assert capture.isOpened()
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 6
        ok, frame = capture.read()
        capture.release()
    assert ok
    # Orange mask is confined to the selected box, and video content survives outside it.
    assert int(frame[6, 6, 2]) > int(frame[6, 6, 0])
    assert int(frame[14, 20, 2]) < int(frame[6, 6, 2])
