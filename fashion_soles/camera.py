"""Live camera feed with slowly refreshed clothing overlays.

The browser draws the camera stream onto a canvas, turned by ``rotation`` so a
sideways camera shows an upright person, mirrored like a mirror. It requests the
widest picture the camera offers (full resolution, minimum zoom) and sends
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
      <video class="live-camera-source" autoplay muted playsinline></video>
      <div class="live-camera-stage piece-preview">
        <canvas class="piece-preview-base"></canvas>
        <img class="live-camera-overlay" alt="">
        <div class="live-camera-highlights"></div>
      </div>
      <div class="live-camera-bar">
        <label><input type="checkbox" class="live-camera-toggle"> Live analysis</label>
        <button type="button" class="live-camera-capture">Capture now</button>
        <span class="live-camera-status">Starting camera…</span>
      </div>`;
    parentElement.appendChild(root);
    root.state = { sent: null, last: 0, signature: null, stream: null, timer: null, frame: null };
    const video = root.querySelector("video");
    const canvas = root.querySelector("canvas");
    // Ask for the sensor's full resolution: smaller modes are often a crop, i.e. zoomed in.
    navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 3840 }, height: { ideal: 2160 }, resizeMode: "none" }, audio: false })
      .then(async (stream) => {
        root.state.stream = stream;
        video.srcObject = stream;
        const track = stream.getVideoTracks()[0];
        const zoom = track.getCapabilities ? track.getCapabilities().zoom : undefined;
        if (zoom) await track.applyConstraints({ advanced: [{ zoom: zoom.min }] }).catch(() => {});
        const settings = track.getSettings();
        status(`Camera ${settings.width}×${settings.height}` + (zoom ? `, zoom ${settings.zoom}` : ""));
      })
      .catch((error) => status("Camera unavailable: " + error.message));
    // Draw the stream turned upright, capped at 1280 px; detection and overlays use this picture.
    const draw = () => {
      if (video.videoWidth) {
        const turn = ((root.rotation % 360) + 360) % 360;
        const scale = Math.min(1, 1280 / Math.max(video.videoWidth, video.videoHeight));
        const w = Math.round(video.videoWidth * scale), h = Math.round(video.videoHeight * scale);
        const [cw, ch] = turn % 180 ? [h, w] : [w, h];
        if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
        const context = canvas.getContext("2d");
        context.save();
        context.translate(cw / 2, ch / 2);
        context.rotate(turn * Math.PI / 180);
        context.drawImage(video, -w / 2, -h / 2, w, h);
        context.restore();
      }
      root.state.frame = requestAnimationFrame(draw);
    };
    root.state.frame = requestAnimationFrame(draw);
    const capture = () => {
      const state = root.state;
      if (state.sent !== null || !canvas.width) return;
      const scale = Math.min(1, root.maxSide / Math.max(canvas.width, canvas.height));
      const still = document.createElement("canvas");
      still.width = Math.round(canvas.width * scale);
      still.height = Math.round(canvas.height * scale);
      still.getContext("2d").drawImage(canvas, 0, 0, still.width, still.height);
      state.sent = state.last = Date.now();
      status("Analysing…");
      setTriggerValue("frame", { seq: state.sent, image: still.toDataURL("image/jpeg", 0.85) });
    };
    root.querySelector(".live-camera-capture").onclick = capture;
    root.state.timer = setInterval(() => {
      const due = Date.now() - root.state.last >= root.intervalMs;
      if (due && root.querySelector(".live-camera-toggle").checked) capture();
    }, 250);
  }
  root.maxSide = data.max_side;
  root.intervalMs = data.interval_ms;
  root.rotation = data.rotation;
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
    cancelAnimationFrame(root.state.frame);
    if (root.state.stream) root.state.stream.getTracks().forEach((track) => track.stop());
  };
}
"""

_CSS = """
.live-camera-source {position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;}
.live-camera-stage {position:relative; transform:scaleX(-1);}
.live-camera-stage canvas {display:block; width:100%; object-fit:contain; background:#000;}
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
