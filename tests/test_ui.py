from pathlib import Path
import io
from PIL import Image
from streamlit.testing.v1 import AppTest
from fashion_soles.fashion_detector import PieceProposal
from fashion_soles.pipeline import image_fingerprint


def test_demo_edit_and_regenerate(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    next(b for b in app.button if b.label == "Load demo analysis").click().run()
    assert not app.exception
    app.text_input[1].set_value("red")
    next(b for b in app.button if b.label == "Apply corrections").click().run()
    assert app.session_state["analysis"]["garments"][0]["colors"] == ["red"]
    assert any(r["rule_id"] == "color_red" for r in app.session_state["trace"]["matched_rules"])
    next(b for b in app.button if b.label == "Generate poem").click().run()
    first = app.session_state["poem"]["lines"]
    next(b for b in app.button if b.label == "Regenerate poem").click().run()
    assert app.session_state["poem"]["lines"] != first
    assert not app.exception


def test_piece_checkboxes_control_preview(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    upload = io.BytesIO()
    Image.new("RGB", (40, 20), (100, 100, 100)).save(upload, format="PNG")
    monkeypatch.setattr("streamlit.file_uploader", lambda *args, **kwargs: upload)
    mask = Image.new("L", (40, 20), 255)
    second_mask = Image.new("L", (40, 20), 255)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["image_fingerprint"] = image_fingerprint(upload.getvalue())
    app.session_state["piece_proposals"] = [
        PieceProposal("jacket", .9, (0, 0, 20, 20), mask, "Provisional intersection", ("#112233", "#445566")),
        PieceProposal("shoe", .8, (20, 0, 40, 20), second_mask, "Provisional intersection", ("#778899",)),
        PieceProposal("bag, wallet", .7, (0, 0, 40, 20), None, "Box only"),
    ]
    app.run()
    assert not app.exception
    boxes = [box for box in app.checkbox if box.label.startswith("Show ")]
    assert len(boxes) == 2
    assert all("bag" not in box.label for box in boxes)
    assert all(box.value for box in boxes)
    assert any('background-color:#112233' in block.value for block in app.markdown)
    boxes[0].uncheck().run()
    assert app.session_state["piece_selected_0"] is False
    assert app.session_state["piece_selected_1"] is True
    assert not app.exception
