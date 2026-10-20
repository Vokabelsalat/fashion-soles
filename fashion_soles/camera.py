"""Live camera feed with slowly refreshed clothing overlays.

The browser shows the camera stream itself (mirrored, like a mirror) and sends
a downscaled still to Python only after the previous one has been analysed, so
a slow detector never builds up a queue. Python answers with transparent
overlay layers that are laid over the live video. Frames go to the local
Streamlit server only and are not stored.
"""
from __future__ import annotations

from typing import Any, Callable

import streamlit as st

_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let root = parentElement.querySelector(".live-camera");
  const status = (text) => { root.querySelector(".live-camera-status").textContent = text; };
  if (!root) {
    root = document.createElement("div");
    root.className = "live-camera";
    root.innerHTML = `
      <div class="live-camera-stage piece-preview">
        <video class="piece-preview-base" autoplay muted playsinline></video>
        <img class="live-camera-overlay" alt="">
        <div class="live-camera-highlights"></div>
      </div>
      <div class="live-camera-bar">
        <label><input type="checkbox" class="live-camera-toggle"> Live analysis</label>
        <button type="button" class="live-camera-capture">Capture now</button>
        <span class="live-camera-status">Starting camera…</span>
      </div>`;
    parentElement.appendChild(root);
    root.state = { sent: null, last: 0, signature: null, stream: null, timer: null };
    const video = root.querySelector("video");
    navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      .then((stream) => { root.state.stream = stream; video.srcObject = stream; status("Camera on"); })
      .catch((error) => status("Camera unavailable: " + error.message));
    const capture = () => {
      const state = root.state;
      if (state.sent !== null || !video.videoWidth) return;
      const scale = Math.min(1, root.maxSide / Math.max(video.videoWidth, video.videoHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      state.sent = state.last = Date.now();
      status("Analysing…");
      setTriggerValue("frame", { seq: state.sent, image: canvas.toDataURL("image/jpeg", 0.85) });
    };
    root.querySelector(".live-camera-capture").onclick = capture;
    root.state.timer = setInterval(() => {
      const due = Date.now() - root.state.last >= root.intervalMs;
      if (due && root.querySelector(".live-camera-toggle").checked) capture();
    }, 250);
  }
  root.maxSide = data.max_side;
  root.intervalMs = data.interval_ms;
  // Python echoes the frame it analysed; only then may the next still be sent.
  if (root.state.sent !== null && data.analyzed === root.state.sent) {
    root.state.sent = null;
    status(data.error ? "Analysis failed: " + data.error : "Updated " + new Date().toLocaleTimeString());
  }
  if (data.signature !== root.state.signature) {
    root.state.signature = data.signature;
    const overlay = root.querySelector(".live-camera-overlay");
    overlay.style.display = data.overlay ? "" : "none";
    if (data.overlay) overlay.src = data.overlay;
    root.querySelector(".live-camera-highlights").innerHTML = data.highlights
      .map(([index, src]) => `<img class="piece-highlight piece-highlight-${index}" src="${src}" alt="">`)
      .join("");
  }
  return () => {
    clearInterval(root.state.timer);
    if (root.state.stream) root.state.stream.getTracks().forEach((track) => track.stop());
  };
}
"""

_CSS = """
.live-camera-stage {position:relative; transform:scaleX(-1);}
.live-camera-stage video {display:block; width:100%; object-fit:contain; background:#000;}
.live-camera-stage .live-camera-overlay, .live-camera-stage .live-camera-highlights,
.live-camera-stage .live-camera-highlights img {position:absolute; inset:0; width:100%; height:100%;
  object-fit:contain; pointer-events:none;}
.live-camera-bar {display:flex; align-items:center; gap:.8rem; margin-top:.4rem; font-size:.85rem; color:#D9D2C3;}
.live-camera-bar input {accent-color:#B08D57;}
.live-camera-bar button {background:#1A1815; border:1px solid #4A4134; color:#D9D2C3; border-radius:2px;
  padding:.2rem .6rem; cursor:pointer;}
.live-camera-bar button:hover {border-color:#B08D57; color:#B08D57;}
.live-camera-status {margin-left:auto; color:#9A9284; font-size:.78rem;}
"""


def live_camera(data: dict[str, Any], on_frame: Callable[[dict[str, Any]], None], key: str) -> None:
    """Mount the camera; ``on_frame`` receives ``{"seq", "image"}`` with a JPEG data URL."""
    def changed() -> None:
        on_frame(st.session_state[key]["frame"])

    # Registered per mount: each Streamlit runtime has its own registry, and an identical re-registration is silent.
    component = st.components.v2.component("live_camera", js=_JS, css=_CSS, isolate_styles=False)
    component(key=key, data=data, on_frame_change=changed)
