from __future__ import annotations

import html
from pathlib import Path

import streamlit as st
from PIL import Image, ImageChops

from fashion_soles.fashion_detector import (available_pieces, piece_highlight, piece_thumbnail,
                                            preview_selected_pieces, propose_pieces)
from fashion_soles.color_names import fashion_color_name
from fashion_soles.from_pieces import PieceInput, analysis_from_pieces, mask_outline
from fashion_soles.mapping import map_analysis
from fashion_soles.pipeline import clear_piece_controls, export_trace, image_fingerprint, reset_for_new_image, stream_fingerprint
from fashion_soles.poetry import generate_poem
from fashion_soles.preprocessing import prepare_image
from fashion_soles.palette import color_histogram
from fashion_soles.presentation import (app_header_html, color_grid_html, color_histogram_html, highlight_preview_html,
                                        outfit_palette_html)
from fashion_soles.schemas import MappingTrace, OutfitAnalysis, Poem
from fashion_soles.scrubber import frame_scrubber
from fashion_soles.segmentation import segment_clothes
from fashion_soles.video import (VIDEO_SUFFIXES, key_frame_index, render_annotated_video, sample_video,
                                 select_video_pieces)

st.set_page_config(page_title="Fashion Soles", page_icon="✦", layout="wide")
st.markdown("""<style>
@font-face {font-family:"Cinzel"; src:url("app/static/fonts/Cinzel.ttf") format("truetype");
  font-weight:400 900; font-display:swap;}
/* Dark Souls palette: ash-black stone, bone text, bronze armour as the accent. */
:root {--nav-h:3.6rem; --bg:#0E0D0B; --stone:#1A1815; --stone-hi:#27231D; --line:#4A4134;
  --bone:#D9D2C3; --ash:#9A9284; --bronze:#B08D57;}
html, body, [data-testid="stAppViewContainer"] {background:var(--bg); color:var(--bone);}
header[data-testid="stHeader"] {display:none !important;}
@media (max-width:1100px) {:root {--nav-h:4.6rem;}}
@media (max-width:700px) {:root {--nav-h:0rem;}}
.block-container {max-width:none; padding:calc(var(--nav-h) + .45rem) .75rem .5rem;}
.block-container [data-testid="stVerticalBlock"] {gap:.4rem;}
.piece-preview {position:relative;}
.piece-preview img {display:block; width:100%; object-fit:contain;}
.piece-preview .piece-highlight {position:absolute; inset:0; height:100%; opacity:0;
  transition:opacity .15s; pointer-events:none;}
.piece-preview .piece-preview-base {transition:opacity .15s;}
[class*="st-key-piece_row_"] {border-radius:2px; transition:background .15s;}
[class*="st-key-piece_row_"]:hover {background:var(--stone-hi);}
.st-key-photo_panel .piece-preview img,
.st-key-photo_panel [data-testid="stImage"] img {
  width:100%; max-height:calc(100dvh - var(--nav-h) - 110px); object-fit:contain;
}
.st-key-video_photo_panel .piece-preview img,
.st-key-video_photo_panel [data-testid="stImage"] img,
.st-key-video_photo_panel video {
  width:100%; max-height:calc(100dvh - var(--nav-h) - 160px); object-fit:contain;
}
.st-key-piece_list, .st-key-poem_body {
  max-height:calc(100dvh - var(--nav-h) - 110px); overflow-y:auto;
}
@media (max-width:700px) {
  .st-key-photo_panel .piece-preview img,
  .st-key-photo_panel [data-testid="stImage"] img,
  .st-key-video_photo_panel .piece-preview img,
  .st-key-video_photo_panel [data-testid="stImage"] img,
  .st-key-video_photo_panel video {max-height:55dvh;}
  .st-key-piece_list, .st-key-poem_body {max-height:none; overflow:visible;}
}
h1,h2,h3 {font-family:"Cinzel",Georgia,serif !important; font-weight:400 !important; color:var(--bronze) !important;
  letter-spacing:.04em;}
h1 {font-size:1.9rem !important; margin:0 !important;}
h2,h3 {font-size:1.3rem !important; margin:0 !important;}
p, label, .stMarkdown {color:var(--bone);}
/* Button labels are <p> elements; let them take the button's state color instead of the rule above. */
button p, button span, button div {color:inherit !important;}
div.stButton button, div.stDownloadButton button, [data-testid="stFileUploader"] button,
[data-testid="stPopover"] button {
  background:var(--stone) !important; border:1px solid var(--line) !important;
  color:var(--bone) !important; border-radius:2px !important;
}
div.stButton button:hover, div.stButton button:focus, div.stButton button:active,
div.stDownloadButton button:hover, div.stDownloadButton button:focus, div.stDownloadButton button:active,
[data-testid="stFileUploader"] button:hover, [data-testid="stFileUploader"] button:focus,
[data-testid="stPopover"] button:hover, [data-testid="stPopover"] button:focus,
[data-testid="stPopover"] button:active {
  background:var(--stone-hi) !important; border-color:var(--bronze) !important; color:var(--bronze) !important;
  text-decoration:none;
}
/* Exactly one button is primary at a time: the next step (upload, detect, or generate). */
div.stButton button[kind="primary"], [data-testid="stFileUploader"] button {
  background:var(--bronze) !important; border-color:var(--bronze) !important; color:var(--bg) !important;
  box-shadow:0 0 .7rem rgba(176,141,87,.45);
}
div.stButton button[kind="primary"]:hover, div.stButton button[kind="primary"]:focus,
div.stButton button[kind="primary"]:active,
[data-testid="stFileUploader"] button:hover, [data-testid="stFileUploader"] button:focus,
[data-testid="stFileUploader"] button:active {
  background:var(--bone) !important; border-color:var(--bone) !important; color:var(--bg) !important;
}
div.stButton button:disabled {background:var(--stone) !important; color:var(--ash) !important;
  border-color:var(--line) !important; box-shadow:none; opacity:.6;}
div[data-testid="stFileUploader"] {border-radius:2px;}
[data-testid="stRadioOption"] {background:transparent !important;}
.piece-color-grid {display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:.3rem; width:100%;}
.piece-swatch {height:1.5rem; border:1px solid var(--line); border-radius:2px;}
/* Streamlit pulls each markdown block's successor up by 1rem, which slides the next element over
   the app's own block-level HTML (preview, histograms, palette, poem). Cancel it for those. */
[data-testid="stMarkdownContainer"]:has(.piece-preview, .frame-stack, .color-hist, .outfit-palette, .poem) {
  margin-bottom:0 !important;}
.outfit-palette {display:flex; flex-wrap:wrap; gap:.35rem .8rem; margin:.1rem 0 0 !important;}
.st-key-outfit_palette {padding-bottom:.4rem;}
.outfit-palette-item {display:flex; align-items:center; gap:.3rem; font-size:.9rem;}
.outfit-palette-swatch {display:inline-block; width:1.15rem; height:1.15rem; border:1px solid var(--line); border-radius:2px;}
.color-hist {margin:0 0 .7rem;}
.color-hist figcaption {display:flex; justify-content:space-between; gap:.5rem; font-size:.85rem;
  color:var(--bone); margin-bottom:.3rem;}
.color-hist figcaption span {color:var(--ash); font-size:.75rem;}
.color-hist-plot {display:flex; align-items:flex-end; gap:2px; height:7.5rem; padding-top:1.1rem;
  border-bottom:1px solid var(--line);}
.color-hist-bar {position:relative; flex:1; min-width:3px; border-radius:4px 4px 0 0;
  box-shadow:inset 0 0 0 1px rgba(217,210,195,.18);}
.color-hist-value {position:absolute; bottom:100%; left:50%; transform:translateX(-50%);
  font-size:.68rem; color:var(--ash); white-space:nowrap;}
.color-hist-bar:hover {outline:1px solid var(--bone); outline-offset:1px;}
.color-hist-bar:hover::after {content:attr(data-tip); position:absolute; bottom:calc(100% + .35rem); left:50%;
  transform:translateX(-50%); z-index:5; padding:.2rem .45rem; border:1px solid var(--line); border-radius:2px;
  background:var(--stone); color:var(--bone); font-size:.75rem; white-space:nowrap; pointer-events:none;}
.color-hist-empty {font-size:.8rem; color:var(--ash) !important;}
.poem {font-family:Georgia,serif; font-size:1.08rem; line-height:1.5; margin:.2rem 0 .3rem !important;}
.st-key-poem_body h3 {margin-top:.4rem !important;}
hr {margin:.15rem 0 !important;}
.app-header {position:fixed; top:0; left:0; right:0; z-index:1000; box-sizing:border-box; height:var(--nav-h);
  display:flex; align-items:center; gap:1rem; padding:0 .75rem; overflow:hidden;
  background:var(--bg); border-bottom:1px solid var(--line);}
.app-logo {display:flex; align-items:center; gap:.55rem; flex:none; margin-right: 20px;}
.app-logo-mark {position:relative; width:1.5rem; height:2.3rem;}
.app-logo-print {position:absolute; left:.25rem; top:.18rem; width:1rem; height:1.95rem;
  transform:rotate(-16deg);}
.app-logo-forefoot, .app-logo-heel {position:absolute;
  background:repeating-linear-gradient(0deg, var(--bone) 0 .13rem, transparent .13rem .2rem);}
.app-logo-forefoot {left:0; top:0; width:1rem; height:1.12rem;
  border-radius:58% 48% 42% 46% / 62% 58% 42% 38%;}
.app-logo-heel {left:.16rem; bottom:0; width:.72rem; height:.62rem; border-radius:42% 42% 50% 50% / 36% 36% 64% 64%;}
.app-logo-word {position:relative; font-family:"Cinzel",Georgia,serif; font-size:1.45rem; line-height:1;
  letter-spacing:.06em; color:var(--bone); white-space:nowrap;}
/* The Dark Souls hairline: crosses the wordmark mid-height and fades out past both ends. */
.app-logo-word::after {content:""; position:absolute; left:-.5rem; right:-.7rem; top:46%; height:1px;
  background:linear-gradient(90deg, transparent, var(--bone) 12%, var(--bone) 88%, transparent);
  pointer-events:none;}
.app-intro {flex:1; margin:0 !important; font-size:.84rem; line-height:1.3; color:var(--ash) !important;}
.app-intro strong {font-family:Georgia,serif; font-weight:400; font-style:italic; font-size:.95rem; color:var(--bone);}
.app-privacy {color:var(--bone); border-bottom:1px solid var(--line);}
@media (max-width:700px) {.app-header {position:static; height:auto; padding:0 0 .4rem;
  flex-direction:column; align-items:flex-start; gap:.4rem;}}
</style>""", unsafe_allow_html=True)

image = st.session_state.get("photo")
if image is None:
    st.markdown(app_header_html(), unsafe_allow_html=True)
    st.write("Upload a photo or a short video of one consenting adult. The local models find clothing pieces; the poem is a fictional reading of what is visible.")
    upload = st.file_uploader("Photo: JPG, PNG, WEBP (10 MB). Video: MP4, MOV, WEBM (60 seconds, up to 1 GB).",
                              type=["jpg", "jpeg", "png",
                                    "webp", "mp4", "mov", "webm"],
                              key=f"upload_{st.session_state.get('upload_epoch', 0)}")
    if upload is not None:
        try:
            suffix = Path(getattr(upload, "name", "photo.png")).suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                frames = sample_video(upload, suffix)
                reset_for_new_image(
                    st.session_state, stream_fingerprint(upload))
                st.session_state.video_frames = frames
                st.session_state.video_upload = upload
                st.session_state.video_format = {
                    ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"
                }[suffix]
                st.session_state.video_frame_index = len(frames) // 2
                st.session_state.photo = frames[len(frames) // 2].image
            else:
                raw = upload.getvalue()
                prepared = prepare_image(raw)
                reset_for_new_image(st.session_state, image_fingerprint(raw))
                st.session_state.photo = prepared
            st.rerun()
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
else:
    video_frames = st.session_state.get("video_frames")
    st.markdown(app_header_html(), unsafe_allow_html=True)
    photo_column, pieces_column, poem_column = st.columns(
        [1.08, 1.08, 1], gap="small")
    photo_bar = photo_column.columns(
        [2.6, 1], gap="small", vertical_alignment="center")
    pieces_bar = pieces_column.columns(
        [1.6, 1], gap="small", vertical_alignment="center")
    pieces_bar[0].subheader("Detected pieces")
    if video_frames:
        st.session_state.video_frame_index = st.session_state.pop(
            "pending_video_frame", st.session_state.get("video_frame_index", len(video_frames) // 2))
        frame_index = st.session_state.video_frame_index
        image = video_frames[frame_index].image
    else:
        frame_index = 0
    if photo_bar[1].button("Change file", width="stretch"):
        epoch = st.session_state.get("upload_epoch", 0) + 1
        st.session_state.clear()
        st.session_state.upload_epoch = epoch
        st.rerun()
    detect = pieces_bar[1].button("Detect pieces", width="stretch",
                                  type="secondary" if available_pieces(st.session_state.get("piece_proposals", []))
                                  else "primary")
    if detect:
        with pieces_column, st.spinner("Finding clothing pieces in sampled frames…" if video_frames else
                                       "Finding clothing pieces with the local models…"):
            try:
                if video_frames:
                    frame_proposals, segmentations = [], []
                    for frame in video_frames:
                        seg = segment_clothes(frame.image)
                        segmentations.append(seg)
                        frame_proposals.append(
                            propose_pieces(frame.image, seg.mask,
                                           seg.accessory_mask)
                            if seg.mask is not None or seg.accessory_mask is not None else [])
                    pieces = select_video_pieces(frame_proposals)
                    st.session_state.video_frame_proposals = frame_proposals
                    st.session_state.pop("annotated_video", None)
                    st.session_state.pop("annotated_video_labels", None)
                    st.session_state.piece_proposals = [
                        item.proposal for item in pieces]
                    st.session_state.video_piece_frames = [
                        item.frame_index for item in pieces]
                    key_frame = key_frame_index(segmentations)
                    st.session_state.segmentation = segmentations[key_frame]
                    st.session_state.pending_video_frame = key_frame
                    st.session_state.pop("frame_preview_cache", None)
                else:
                    seg = segment_clothes(image)
                    st.session_state.segmentation = seg
                    st.session_state.piece_proposals = (
                        propose_pieces(image, seg.mask, seg.accessory_mask)
                        if seg.mask is not None or seg.accessory_mask is not None else []
                    )
                clear_piece_controls(st.session_state)
                for key in ("analysis", "trace", "poem", "poem_basis", "variation"):
                    st.session_state.pop(key, None)
                st.rerun()
            except Exception as exc:
                st.error(f"Detection failed: {type(exc).__name__}: {exc}")

    proposals = available_pieces(st.session_state.get("piece_proposals", []))
    selected_pieces = [piece for index, piece in enumerate(proposals)
                       if st.session_state.get(f"piece_selected_{index}", True)]

    def preview_pairs(frame: int) -> list:
        """Pair each selected list index with its piece on a frame, so hovering a row can highlight it.

        Video lists hold one piece per label, matched to the frame's detection of that label.
        """
        selected = [(index, piece) for index, piece in enumerate(proposals)
                    if st.session_state.get(f"piece_selected_{index}", True)]
        if not (video_frames and "video_frame_proposals" in st.session_state):
            return selected
        frame_pieces = {piece.label.strip().lower(): piece
                        for piece in st.session_state.video_frame_proposals[frame]}
        return [(index, frame_pieces[piece.label.strip().lower()]) for index, piece in selected
                if piece.label.strip().lower() in frame_pieces]

    def frame_preview_html(frame_image, frame: int) -> str:
        pairs = preview_pairs(frame)
        return highlight_preview_html(preview_selected_pieces(frame_image, [piece for _, piece in pairs]),
                                      {index: piece_highlight(frame_image, piece) for index, piece in pairs})

    def video_preview_html() -> str:
        """Stack every sampled frame so the scrubber can swap them in the browser while dragging."""
        # Rendering all frames on each rerun is slow, so reuse them until detection or selection changes.
        cache = st.session_state.setdefault("frame_preview_cache", {})
        layers = []
        for frame, sample in enumerate(video_frames):
            key = (frame, tuple(index for index, _ in preview_pairs(frame)))
            if key not in cache:
                cache[key] = frame_preview_html(sample.image, frame)
            hidden = "" if frame == frame_index else ' style="display:none"'
            layers.append(
                f'<div class="frame-layer" data-frame="{frame}"{hidden}>{cache[key]}</div>')
        return '<div class="frame-stack">' + "".join(layers) + "</div>"

    with photo_column:
        with st.container(key="video_photo_panel" if video_frames else "photo_panel", border=False):
            view_options = ("Analyzed frame", "Replay video", "Overlay replay") if video_frames and proposals else (
                "Analyzed frame", "Replay video")
            view = (photo_bar[0].radio("Preview", view_options,
                                       horizontal=True, label_visibility="collapsed", key="video_view")
                    if video_frames else "Analyzed frame")
            if view == "Replay video":
                video_upload = st.session_state.get("video_upload")
                if video_upload is None:
                    st.write(
                        "Choose Change file and upload the clip again to enable replay.")
                else:
                    st.video(video_upload, format=st.session_state.video_format,
                             autoplay=False, width="stretch")
            elif view == "Overlay replay":
                video_upload = st.session_state.get("video_upload")
                if video_upload is None:
                    st.write(
                        "Choose Change file and upload the clip again to enable replay.")
                elif not selected_pieces:
                    st.write("Select at least one piece to show its overlay.")
                else:
                    labels = tuple(
                        sorted({piece.label for piece in selected_pieces}))
                    if st.session_state.get("annotated_video_labels") != labels:
                        st.session_state.pop("annotated_video", None)
                    if "annotated_video" not in st.session_state:
                        with st.spinner("Rendering clothing overlays on the video…"):
                            try:
                                st.session_state.annotated_video = render_annotated_video(
                                    video_upload, Path(
                                        video_upload.name).suffix.lower(), video_frames,
                                    st.session_state.video_frame_proposals, set(labels))
                                st.session_state.annotated_video_labels = labels
                            except (ValueError, RuntimeError) as exc:
                                st.error(str(exc))
                    if "annotated_video" in st.session_state:
                        st.video(st.session_state.annotated_video,
                                 format="video/mp4", width="stretch")
                        st.write(
                            "Boxes use sampled detections; shapes are refreshed on rendered frames. The overlay replay has no audio.")
            else:
                if video_frames:
                    st.markdown(video_preview_html(), unsafe_allow_html=True)
                    frame_scrubber([f"{sample.seconds:.1f}s" for sample in video_frames], frame_index,
                                   lambda frame: st.session_state.update(
                                       video_frame_index=frame),
                                   key="frame_scrubber")
                else:
                    st.markdown(frame_preview_html(
                        image, frame_index), unsafe_allow_html=True)
                if not proposals and "piece_proposals" not in st.session_state:
                    st.write(
                        "Run detection to add clothing shapes and boxes to this image.")
                elif not proposals:
                    seg = st.session_state.get("segmentation")
                    st.write(seg.reason if seg is not None and seg.mask is None else
                             "No pieces had usable masks. Try a clearer photo or check the segmentation setup.")
                with st.expander("Color histograms (debug)"):
                    # All detected pieces on this frame, selected or not, so the view shows what segmentation found.
                    frame_pieces = (available_pieces(st.session_state.video_frame_proposals[frame_index])
                                    if video_frames and "video_frame_proposals" in st.session_state else proposals)
                    clothes = Image.new("L", image.size, 0)
                    for piece in frame_pieces:
                        mask = piece.mask.convert("L")
                        clothes = ImageChops.lighter(clothes, mask if mask.size == image.size else
                                                     mask.resize(image.size, Image.Resampling.NEAREST))
                    mode = st.session_state.get("color_mode") or "basic"
                    st.markdown(color_histogram_html("Whole image", color_histogram(image), mode)
                                + color_histogram_html(f"Clothing pieces ({len(frame_pieces)})",
                                                       color_histogram(image, clothes) if frame_pieces else [], mode),
                                unsafe_allow_html=True)

    def reset_color_names() -> None:
        # Edited names were picked in the old mode; drop them so defaults follow the new list.
        for key in [key for key in st.session_state if key.startswith("piece_colors_")]:
            del st.session_state[key]

    with pieces_column:
        selected_items = []
        color_mode = "basic"
        if proposals:
            color_mode = st.segmented_control(
                "Color names", ["basic", "whimsical"], default="basic", key="color_mode",
                format_func=lambda mode: {
                    "basic": "Basic colors", "whimsical": "Whimsical colors"}[mode],
                on_change=reset_color_names, label_visibility="collapsed") or "basic"
            with st.container(key="piece_list", border=False):
                for index, piece in enumerate(proposals):
                    with st.container(key=f"piece_row_{index}", border=False):
                        row = st.columns(
                            [.38, .68, 1.95, .55, 1.15], gap="small", vertical_alignment="center")
                        enabled = row[0].checkbox(f"Use {piece.label} {index + 1}", value=True,
                                                  key=f"piece_selected_{index}", label_visibility="collapsed")
                        source_image = (video_frames[st.session_state.video_piece_frames[index]].image
                                        if video_frames else image)
                        row[1].image(piece_thumbnail(
                            source_image, piece.mask), width="stretch")
                        display_name = st.session_state.get(
                            f"piece_name_{index}", piece.label)
                        default_color = (fashion_color_name(piece.colors[0], color_mode)
                                         if piece.colors else "unknown")
                        display_color = st.session_state.get(
                            f"piece_colors_{index}", default_color)
                        row[2].write(
                            f"{display_name} · {display_color} · {piece.score:.0%}")
                        row[3].markdown(color_grid_html(
                            piece.colors), unsafe_allow_html=True)
                        with row[4].popover("Edit"):
                            name = st.text_input(
                                "Piece name", piece.label, key=f"piece_name_{index}")
                            colors_text = st.text_input("Color name", default_color,
                                                        key=f"piece_colors_{index}")
                            outline_options = ["vertical", "broad", "balanced"]
                            default_outline = mask_outline(piece.mask)
                            outline = st.selectbox("Visible outline", outline_options,
                                                   index=outline_options.index(
                                                       default_outline)
                                                   if default_outline in outline_options else 2,
                                                   key=f"piece_outline_{index}")
                        if enabled:
                            selected_items.append(PieceInput(piece, name,
                                                             (colors_text.strip(),) if colors_text.strip() else (), outline, source_image))
        elif "piece_proposals" in st.session_state:
            st.write("No pieces with usable masks were found.")
        else:
            st.write("Detected clothing pieces will appear here.")

    analysis = analysis_from_pieces(
        selected_items, color_mode) if selected_items else None
    trace = map_analysis(analysis) if analysis else None
    basis = analysis.model_dump_json() if analysis else None
    if analysis is None:
        st.session_state.pop("poem", None)

    with poem_column:
        st.subheader("Poem")
        if analysis is not None and analysis.overall.palette:
            with st.container(key="outfit_palette", border=False):
                st.markdown(outfit_palette_html(
                    analysis.overall.palette, color_mode), unsafe_allow_html=True)
        action = st.columns([1.2, 1], gap="small", vertical_alignment="center")
        generate = action[0].button("Generate poem", type="primary" if proposals else "secondary",
                                    disabled=analysis is None,
                                    width="stretch")
        with action[1].popover("Poem options"):
            tone = st.selectbox(
                "Tone", ["lyrical", "minimal", "fragmentary"], key="poem_tone")
            length = st.slider("Lines", 8, 16, 12, key="poem_length")
        # After the first poem, any change to its inputs rewrites it; the button asks for a new variation.
        poem_inputs = (basis, tone, length, color_mode)
        regenerate = (st.session_state.get("poem_auto") and analysis is not None
                      and ("poem" not in st.session_state or st.session_state.get("poem_inputs") != poem_inputs))
        if (generate or regenerate) and analysis is not None and trace is not None:
            if generate:
                variation = st.session_state.get("variation", 0)
            elif st.session_state.get("poem_basis") == basis:
                # Only tone, length, or mode changed.
                variation = st.session_state.get("poem_variation", 0)
            else:
                variation = 0
            try:
                poem = generate_poem(analysis, trace, length=length, tone=tone, variation=variation,
                                     color_mode=color_mode)
                st.session_state.analysis = analysis.model_dump()
                st.session_state.trace = trace.model_dump()
                st.session_state.poem = poem.model_dump()
                st.session_state.poem_basis = basis
                st.session_state.poem_inputs = poem_inputs
                st.session_state.poem_variation = variation
                st.session_state.variation = variation + 1
                st.session_state.poem_auto = True
            except Exception as exc:
                st.error(f"Poem generation failed: {exc}")

        with st.container(key="poem_body", border=False):
            if st.session_state.get("poem"):
                poem = Poem.model_validate(st.session_state.poem)
                st.subheader(poem.title)
                st.markdown('<div class="poem">' + "<br>".join(html.escape(line) for line in poem.lines) + "</div>",
                            unsafe_allow_html=True)
                with st.expander("Poem details and downloads"):
                    st.write("Motifs:", ", ".join(
                        trace.dominant_motifs) or "none")
                    poem_text = poem.title + "\n\n" + \
                        "\n".join(poem.lines) + "\n"
                    st.download_button("Download poem", poem_text, file_name="fashion-soles-poem.txt",
                                       mime="text/plain")
                    seg = st.session_state.get("segmentation")
                    st.download_button("Download analysis", export_trace(
                        analysis, trace, poem, mode="template",
                        segmentation_status=seg.status if seg else "not run", pieces=selected_pieces,
                        video_samples=[frame.seconds for frame in video_frames] if video_frames else None),
                        file_name="fashion-soles-trace.json", mime="application/json")
            else:
                st.write("Select pieces and generate a poem to see it here.")
