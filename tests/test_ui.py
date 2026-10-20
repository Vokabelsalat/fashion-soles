from pathlib import Path
import io

from PIL import Image
from streamlit.testing.v1 import AppTest

from fashion_soles.fashion_detector import PieceProposal

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_upload_area_disappears_after_photo_is_loaded(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
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
    assert any(button.label == "Change photo" for button in app.button)


def test_selected_pieces_feed_poem_and_edits_reset_it(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
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
    assert app.session_state["poem"]["title"].startswith("Demo poem:")
    next(box for box in app.checkbox if box.label.startswith("Use jacket")).uncheck().run()
    assert not app.exception
    assert "poem" not in app.session_state
    next(button for button in app.button if button.label == "Generate poem").click().run()
    assert [garment["category"] for garment in app.session_state["analysis"]["garments"]] == ["shoe"]
    next(button for button in app.button if button.label == "Change photo").click().run()
    assert "photo" not in app.session_state
    assert not app.exception
