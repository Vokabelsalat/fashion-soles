"""A frame slider that swaps the preview while dragging, not only on release.

Streamlit sliders report their value on release. This component shows the
pre-rendered frame layers (``.frame-stack .frame-layer[data-frame=N]``) in the
browser on every ``input`` event and sends the frame to Python on ``change``.
"""
from __future__ import annotations

from typing import Callable

import streamlit as st

_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let root = parentElement.querySelector(".frame-scrubber");
  if (!root) {
    root = document.createElement("div");
    root.className = "frame-scrubber";
    root.innerHTML = '<input type="range" min="0" step="1" aria-label="Video frame"><output></output>';
    parentElement.appendChild(root);
  }
  const input = root.querySelector("input");
  const output = root.querySelector("output");
  input.max = String(data.labels.length - 1);
  input.value = String(data.value);
  output.textContent = data.labels[data.value];
  input.oninput = () => {
    const index = Number(input.value);
    output.textContent = data.labels[index];
    document.querySelectorAll(".frame-stack .frame-layer").forEach((layer) => {
      layer.style.display = layer.dataset.frame === String(index) ? "" : "none";
    });
  };
  // A trigger (not state) so returning to a frame Python moved away from still reruns.
  input.onchange = () => setTriggerValue("frame", Number(input.value));
}
"""

_CSS = """
.frame-scrubber {display:flex; align-items:center; gap:.6rem;}
.frame-scrubber input {flex:1; accent-color:#B08D57; cursor:pointer;}
.frame-scrubber output {min-width:3rem; text-align:right; font-size:.85rem; color:#9A9284;}
"""


def frame_scrubber(labels: list[str], value: int, on_change: Callable[[int], None], key: str) -> None:
    """Mount the slider; ``on_change`` receives the frame index when the drag ends."""
    def changed() -> None:
        on_change(int(st.session_state[key]["frame"]))

    # Registered per mount: each Streamlit runtime has its own registry, and an identical re-registration is silent.
    component = st.components.v2.component("frame_scrubber", js=_JS, css=_CSS, isolate_styles=False)
    component(key=key, data={"labels": labels, "value": value}, on_frame_change=changed)
