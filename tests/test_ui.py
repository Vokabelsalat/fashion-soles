from pathlib import Path
import io

from PIL import Image
from PIL import ImageDraw
from streamlit.testing.v1 import AppTest

from fashion_soles.fashion_detector import PieceProposal

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_upload_area_disappears_after_photo_is_loaded(monkeypatch):
    upload = io.BytesIO()
    Image.new("RGB", (40, 20), "red").save(upload, format="PNG")
    calls = []

    def fake_uploader(*args, **kwargs):
        calls.append(True)
        return upload

    monkeypatch.setattr("streamlit.file_uploader", fake_uploader)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert app.session_state["photo"].size == (40, 20)
    assert len(calls) == 1
    assert any(button.label == "Change file" for button in app.button)


def test_selected_pieces_feed_poem_and_edits_reset_it(monkeypatch):
    image = Image.new("RGB", (40, 20), "white")
    mask = Image.new("L", image.size, 255)
    app = AppTest.from_file(APP)
    app.session_state["photo"] = image
    app.session_state["piece_proposals"] = [
        PieceProposal("jacket", .9, (0, 0, 20, 20), mask, "Provisional intersection", ("#FF0000",)),
        PieceProposal("shoe", .8, (20, 0, 40, 20), mask, "Provisional intersection", ("#0000FF",)),
        PieceProposal("bag, wallet", .7, (0, 0, 40, 20), None, "Box only"),
    ]
    app.run()
    assert not app.exception
    boxes = [box for box in app.checkbox if box.label.startswith("Use ")]
    assert len(boxes) == 2
    buttons = {button.label: button for button in app.button}
    assert buttons["Generate poem"].proto.type == "primary" and buttons["Detect pieces"].proto.type == "secondary"
    assert any("piece-highlight-0" in item.value and "piece-highlight-1" in item.value
               for item in app.markdown)
    assert all(box.value for box in boxes)
    next(button for button in app.button if button.label == "Generate poem").click().run()
    assert not app.exception
    assert [garment["category"] for garment in app.session_state["analysis"]["garments"]] == ["jacket", "shoe"]
    assert app.session_state["poem"]["title"]
    next(box for box in app.checkbox if box.label.startswith("Use jacket")).uncheck().run()
    assert not app.exception
    assert "poem" not in app.session_state
    next(button for button in app.button if button.label == "Generate poem").click().run()
    assert [garment["category"] for garment in app.session_state["analysis"]["garments"]] == ["shoe"]
    next(button for button in app.button if button.label == "Change file").click().run()
    assert "photo" not in app.session_state
    assert not app.exception


def test_uploaded_video_samples_feed_one_editable_poem(monkeypatch):
    from fashion_soles.segmentation import Segmentation
    from fashion_soles.video import VideoFrame

    upload = io.BytesIO(b"video bytes")
    upload.name = "outfit.mp4"
    frames = [VideoFrame(Image.new("RGB", (40, 20), "red"), 0.0),
              VideoFrame(Image.new("RGB", (40, 20), "blue"), 1.0)]
    mask = Image.new("L", (40, 20), 255)
    uploads = iter((upload, None))
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: next(uploads, None))
    monkeypatch.setattr("fashion_soles.video.sample_video", lambda data, suffix: frames)
    video_calls = []
    monkeypatch.setattr("streamlit.video", lambda *args, **kwargs: video_calls.append((args, kwargs)))
    monkeypatch.setattr("fashion_soles.segmentation.segment_clothes",
                        lambda image: Segmentation("Ready", mask=mask))

    def pieces(image, clothes_mask, accessory_mask):
        if image.getpixel((0, 0)) == (255, 0, 0):
            return [PieceProposal("shirt", .7, (0, 0, 40, 20), mask,
                                  "Provisional intersection", ("#FFFFFF",))]
        return [PieceProposal("shirt", .9, (0, 0, 40, 20), mask,
                              "Provisional intersection", ("#FFFFFF",)),
                PieceProposal("shoe", .8, (0, 0, 40, 20), mask,
                              "Provisional intersection", ("#FFFFFF",))]

    monkeypatch.setattr("fashion_soles.fashion_detector.propose_pieces", pieces)
    overlay_calls = []
    monkeypatch.setattr("fashion_soles.video.render_annotated_video",
                        lambda data, suffix, frames, proposals, labels: overlay_calls.append(labels) or b"rendered")
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert len(app.session_state["video_frames"]) == 2
    assert any(slider.label == "Video frame" for slider in app.select_slider)
    assert next(button for button in app.button if button.label == "Detect pieces").proto.type == "primary"
    assert next(button for button in app.button if button.label == "Generate poem").proto.type == "secondary"
    assert not video_calls
    next(radio for radio in app.radio if radio.label == "Preview").set_value("Replay video").run()
    assert not app.exception
    assert video_calls[-1][0][0] is upload
    assert video_calls[-1][1]["format"] == "video/mp4"
    next(radio for radio in app.radio if radio.label == "Preview").set_value("Analyzed frame").run()
    assert not app.exception
    next(button for button in app.button if button.label == "Detect pieces").click().run()
    assert not app.exception
    assert [piece.label for piece in app.session_state["piece_proposals"]] == ["shirt", "shoe"]
    assert app.session_state["video_piece_frames"] == [1, 1]
    next(radio for radio in app.radio if radio.label == "Preview").set_value("Overlay replay").run()
    assert not app.exception
    assert overlay_calls == [{"shirt", "shoe"}]
    assert video_calls[-1][0][0] == b"rendered"
    app.run()
    assert overlay_calls == [{"shirt", "shoe"}]
    next(box for box in app.checkbox if box.label.startswith("Use shoe")).uncheck().run()
    assert overlay_calls[-1] == {"shirt"}
    next(box for box in app.checkbox if box.label.startswith("Use shoe")).check().run()
    next(button for button in app.button if button.label == "Generate poem").click().run()
    assert not app.exception
    assert [item["name"] for item in app.session_state["analysis"]["garments"]] == ["shirt", "shoe"]
    next(radio for radio in app.radio if radio.label == "Preview").set_value("Analyzed frame").run()
    assert app.session_state["video_frame_index"] == 0
    next(slider for slider in app.select_slider if slider.label == "Video frame").set_value(1).run()
    assert app.session_state["video_frame_index"] == 1
    next(slider for slider in app.select_slider if slider.label == "Video frame").set_value(0).run()
    assert not app.exception
    assert "poem" in app.session_state
    next(button for button in app.button if button.label == "Change file").click().run()
    assert "video_frames" not in app.session_state
    assert "video_upload" not in app.session_state


def test_previous_video_session_can_request_replay_without_source():
    from fashion_soles.video import VideoFrame

    image = Image.new("RGB", (40, 20), "white")
    app = AppTest.from_file(APP)
    app.session_state["photo"] = image
    app.session_state["video_frames"] = [VideoFrame(image, 0.0)]
    app.session_state["video_frame_index"] = 0
    app.run()
    next(radio for radio in app.radio if radio.label == "Preview").set_value("Replay video").run()
    assert not app.exception
    assert any("upload the clip again" in item.value for item in app.markdown)


def test_selected_masks_update_visible_palette_and_poem():
    image = Image.new("RGB", (40, 20), "#FF0000")
    ImageDraw.Draw(image).rectangle((30, 0, 39, 19), fill="#0000FF")
    red_mask = Image.new("L", image.size, 0)
    blue_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(red_mask).rectangle((0, 0, 29, 19), fill=255)
    ImageDraw.Draw(blue_mask).rectangle((30, 0, 39, 19), fill=255)
    app = AppTest.from_file(APP)
    app.session_state["photo"] = image
    app.session_state["piece_proposals"] = [
        PieceProposal("shirt", .9, (0, 0, 30, 20), red_mask, "intersection", ("#FF0000",)),
        PieceProposal("shoe", .9, (30, 0, 40, 20), blue_mask, "intersection", ("#0000FF",)),
    ]
    app.run()
    assert not app.exception
    assert any("Primary: Red" in item.value and "Secondary: Blue" in item.value for item in app.markdown)
    app.segmented_control(key="color_mode").set_value("whimsical").run()
    assert not app.exception
    assert app.session_state["color_mode"] == "whimsical"
    app.segmented_control(key="color_mode").set_value("basic").run()
    next(box for box in app.checkbox if box.label.startswith("Use shirt")).uncheck().run()
    assert not app.exception
    assert any("Primary: Blue" in item.value and "Secondary:" not in item.value for item in app.markdown)
    next(button for button in app.button if button.label == "Generate poem").click().run()
    assert "blue holds the visible pieces" in app.session_state["poem"]["lines"][6]
