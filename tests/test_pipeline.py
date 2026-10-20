import io
import pytest
from PIL import Image
from fashion_soles.pipeline import clear_piece_controls, demo_run, reset_for_new_image, stream_fingerprint
from fashion_soles.preprocessing import prepare_image


def test_demo_trace_and_different_poems():
    first = demo_run(variation=0)
    second = demo_run(variation=1)
    assert first["mode"] == "template"
    assert first["mapping"] == second["mapping"]
    assert first["poem"]["lines"] != second["poem"]["lines"]
    assert len(first["poem"]["lines"]) == 12


def test_invalid_image_and_reset():
    with pytest.raises(ValueError):
        prepare_image(b"not an image")
    with pytest.raises(ValueError):
        prepare_image(b"x" * (10 * 1024 * 1024 + 1))
    state = {"image_fingerprint": "old", "analysis": {}, "trace": {}, "poem": {}}
    assert reset_for_new_image(state, "new")
    assert "analysis" not in state and "poem" not in state
    assert not reset_for_new_image(state, "new")


def test_image_is_rgb_and_metadata_free():
    image = Image.new("RGB", (24, 24), "red")
    output = io.BytesIO()
    image.save(output, format="PNG")
    clean = prepare_image(output.getvalue())
    assert clean.mode == "RGB"
    assert clean.info == {}


def test_piece_controls_reset_for_new_detection_or_image():
    state = {"image_fingerprint": "old", "piece_selected_0": False,
             "piece_selected_1": True, "piece_name_0": "coat", "analysis": {}}
    clear_piece_controls(state)
    assert "piece_selected_0" not in state
    assert "piece_selected_1" not in state
    assert "piece_name_0" not in state
    state["piece_selected_0"] = False
    assert reset_for_new_image(state, "new")
    assert "piece_selected_0" not in state


def test_new_upload_clears_sampled_video_state():
    state = {"image_fingerprint": "old", "video_frames": [object()],
             "video_upload": object(), "video_format": "video/mp4", "video_view": "Replay video",
             "video_frame_proposals": [[]], "video_piece_frames": [0],
             "video_frame_index": 0, "photo": object()}
    assert reset_for_new_image(state, "new")
    assert not any(key.startswith("video_") for key in state)
    assert "photo" not in state


def test_video_fingerprint_reads_in_chunks_and_resets_position():
    source = io.BytesIO(b"video" * 100)
    source.seek(9)
    assert stream_fingerprint(source) == __import__("hashlib").sha256(b"video" * 100).hexdigest()
    assert source.tell() == 0
