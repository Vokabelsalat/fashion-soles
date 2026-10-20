from pathlib import Path
from streamlit.testing.v1 import AppTest


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
