from pathlib import Path
import io

from PIL import Image
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
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert len(app.session_state["video_frames"]) == 2
    assert any(select.label == "Video frame" for select in app.selectbox)
    next(button for button in app.button if button.label == "Detect pieces").click().run()
    assert not app.exception
    assert [piece.label for piece in app.session_state["piece_proposals"]] == ["shirt", "shoe"]
    assert app.session_state["video_piece_frames"] == [1, 1]
    next(button for button in app.button if button.label == "Generate poem").click().run()
    assert not app.exception
    assert [item["name"] for item in app.session_state["analysis"]["garments"]] == ["shirt", "shoe"]
    next(select for select in app.selectbox if select.label == "Video frame").set_value(0).run()
    assert not app.exception
    assert "poem" in app.session_state
    next(button for button in app.button if button.label == "Change file").click().run()
    assert "video_frames" not in app.session_state
