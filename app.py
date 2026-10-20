from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from fashion_soles.fashion_detector import (available_pieces, piece_highlight, piece_thumbnail,
                                            preview_selected_pieces, propose_pieces)
from fashion_soles.from_pieces import PieceInput, analysis_from_pieces, color_name, mask_outline
from fashion_soles.mapping import map_analysis
from fashion_soles.pipeline import clear_piece_controls, export_trace, image_fingerprint, reset_for_new_image, stream_fingerprint
from fashion_soles.poetry import generate_poem
from fashion_soles.preprocessing import prepare_image
from fashion_soles.presentation import app_header_html, color_grid_html, highlight_preview_html, outfit_palette_html
from fashion_soles.schemas import MappingTrace, OutfitAnalysis, Poem
from fashion_soles.segmentation import segment_clothes
from fashion_soles.video import (VIDEO_SUFFIXES, key_frame_index, render_annotated_video, sample_video,
                                 select_video_pieces)

st.set_page_config(page_title="Fashion Soles", page_icon="✦", layout="wide")
st.markdown("""<style>
html, body, [data-testid="stAppViewContainer"] {background:#f5f2eb; color:#292720;}
header[data-testid="stHeader"] {display:none !important;}
.block-container {max-width:none; padding:.35rem .75rem .5rem;}
.block-container [data-testid="stVerticalBlock"] {gap:.4rem;}
.piece-preview {position:relative;}
.piece-preview img {display:block; width:100%; object-fit:contain;}
.piece-preview .piece-highlight {position:absolute; inset:0; height:100%; opacity:0;
  transition:opacity .15s; pointer-events:none;}
.piece-preview .piece-preview-base {transition:opacity .15s;}
[class*="st-key-piece_row_"] {border-radius:2px; transition:background .15s;}
[class*="st-key-piece_row_"]:hover {background:#e9e2d4;}
.st-key-photo_panel .piece-preview img,
.st-key-photo_panel [data-testid="stImage"] img {
  width:100%; max-height:calc(100dvh - 115px); object-fit:contain;
}
.st-key-video_photo_panel .piece-preview img,
.st-key-video_photo_panel [data-testid="stImage"] img,
.st-key-video_photo_panel video {
  width:100%; max-height:calc(100dvh - 155px); object-fit:contain;
}
.st-key-piece_list, .st-key-poem_body {
  max-height:calc(100dvh - 180px); overflow-y:auto;
}
@media (max-width:700px) {
  .st-key-photo_panel .piece-preview img,
  .st-key-photo_panel [data-testid="stImage"] img,
  .st-key-video_photo_panel .piece-preview img,
  .st-key-video_photo_panel [data-testid="stImage"] img,
  .st-key-video_photo_panel video {max-height:55dvh;}
  .st-key-piece_list, .st-key-poem_body {max-height:none; overflow:visible;}
}
h1,h2,h3 {font-family:Georgia,serif !important; font-weight:400 !important; color:#292720 !important;}
h1 {font-size:1.9rem !important; margin:0 !important;}
h2,h3 {font-size:1.3rem !important; margin:0 !important;}
p, label, .stMarkdown {color:#292720;}
div.stButton button, div.stDownloadButton button, [data-testid="stFileUploader"] button,
[data-testid="stPopover"] button {
  background:#f5f2eb !important; border:1px solid #292720 !important;
  color:#292720 !important; border-radius:2px !important;
}
div.stButton button:hover, div.stButton button:focus,
div.stDownloadButton button:hover, div.stDownloadButton button:focus,
[data-testid="stFileUploader"] button:hover, [data-testid="stFileUploader"] button:focus,
[data-testid="stPopover"] button:hover, [data-testid="stPopover"] button:focus {
  background:#f5f2eb !important; border-color:#292720 !important; color:#292720 !important;
  text-decoration:underline;
}
div[data-testid="stFileUploader"] {border-radius:2px;}
[data-testid="stRadioOption"] {background:transparent !important;}
.piece-color-grid {display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:.3rem; width:100%;}
.piece-swatch {height:1.5rem; border:1px solid #292720; border-radius:2px;}
.outfit-palette {display:flex; flex-wrap:wrap; gap:.35rem .8rem; margin:.1rem 0 .3rem;}
.outfit-palette-item {display:flex; align-items:center; gap:.3rem; font-size:.9rem;}
.outfit-palette-swatch {display:inline-block; width:1.15rem; height:1.15rem; border:1px solid #292720; border-radius:2px;}
.poem {font-family:Georgia,serif; font-size:1.08rem; line-height:1.42; white-space:pre-line;}
hr {margin:.15rem 0 !important;}
.app-header {display:flex; align-items:center; gap:1rem; min-width:0;}
.app-logo {display:flex; align-items:center; gap:.55rem; flex:none;}
.app-logo-mark {position:relative; width:2.7rem; height:2.7rem; border-radius:50%; background:#292720;}
.app-logo-sole {position:absolute; left:.95rem; top:.55rem; width:.8rem; height:1.65rem; background:#f5f2eb;
  border-radius:50% 50% 45% 45% / 62% 62% 38% 38%; transform:rotate(-18deg);}
.app-logo-sole::after {content:""; position:absolute; left:.12rem; right:.12rem; top:55%; height:2px;
  background:#292720;}
.app-logo-soul {position:absolute; right:.35rem; top:.3rem; width:.45rem; height:.45rem; border-radius:50%;
  background:#E1603D; box-shadow:0 0 .45rem .1rem rgba(225,96,61,.75);}
.app-logo-word {font-family:Georgia,serif; font-size:1.05rem; line-height:1; letter-spacing:.04em;
  text-transform:uppercase;}
.app-intro {margin:0 !important; font-size:.84rem; line-height:1.3; max-width:52rem;}
.app-intro strong {font-family:Georgia,serif; font-weight:400; font-style:italic; font-size:.95rem;}
.app-privacy {border-bottom:2px solid #E1603D;}
@media (max-width:700px) {.app-header {flex-direction:column; align-items:flex-start; gap:.4rem;}}
</style>""", unsafe_allow_html=True)

image = st.session_state.get("photo")
if image is None:
    st.markdown(app_header_html(), unsafe_allow_html=True)
    st.write("Upload a photo or a short video of one consenting adult. The local models find clothing pieces; the poem is a fictional reading of what is visible.")
    upload = st.file_uploader("Photo: JPG, PNG, WEBP (10 MB). Video: MP4, MOV, WEBM (60 seconds, up to 1 GB).",
                              type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "webm"],
                              key=f"upload_{st.session_state.get('upload_epoch', 0)}")
    if upload is not None:
        try:
            suffix = Path(getattr(upload, "name", "photo.png")).suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                frames = sample_video(upload, suffix)
                reset_for_new_image(st.session_state, stream_fingerprint(upload))
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
    header = st.columns([4.6, 1.4, 1, 1], gap="small", vertical_alignment="center") if video_frames else st.columns(
        [6, 1, 1], gap="small", vertical_alignment="center")
    header[0].markdown(app_header_html(), unsafe_allow_html=True)
    if video_frames:
        # A widget's key can only be set before it is drawn, so detection leaves the key frame pending.
        if "pending_video_frame" in st.session_state:
            st.session_state.video_frame_index = st.session_state.pop("pending_video_frame")
        frame_index = header[1].selectbox("Video frame", range(len(video_frames)),
            format_func=lambda index: f"Frame {index + 1} · {video_frames[index].seconds:.1f}s",
            key="video_frame_index", label_visibility="collapsed")
        image = video_frames[frame_index].image
        change_button, detect_button = header[2], header[3]
    else:
        frame_index = 0
        change_button, detect_button = header[1], header[2]
    if change_button.button("Change file", width="stretch"):
        epoch = st.session_state.get("upload_epoch", 0) + 1
        st.session_state.clear()
        st.session_state.upload_epoch = epoch
        st.rerun()
    detect = detect_button.button("Detect pieces", type="primary", width="stretch")
    if detect:
        with st.spinner("Finding clothing pieces in sampled frames…" if video_frames else
                        "Finding clothing pieces with the local models…"):
            try:
                if video_frames:
                    frame_proposals, segmentations = [], []
                    for frame in video_frames:
                        seg = segment_clothes(frame.image)
                        segmentations.append(seg)
                        frame_proposals.append(
                            propose_pieces(frame.image, seg.mask, seg.accessory_mask)
                            if seg.mask is not None or seg.accessory_mask is not None else [])
                    pieces = select_video_pieces(frame_proposals)
                    st.session_state.video_frame_proposals = frame_proposals
                    st.session_state.pop("annotated_video", None)
                    st.session_state.pop("annotated_video_labels", None)
                    st.session_state.piece_proposals = [item.proposal for item in pieces]
                    st.session_state.video_piece_frames = [item.frame_index for item in pieces]
                    key_frame = key_frame_index(segmentations)
                    st.session_state.segmentation = segmentations[key_frame]
                    st.session_state.pending_video_frame = key_frame
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
    # Pair each list index with the piece drawn on the previewed frame, so hovering
    # a row can highlight its shape. Video lists hold one piece per label.
    if video_frames and "video_frame_proposals" in st.session_state:
        frame_pieces = {piece.label.strip().lower(): piece
                        for piece in st.session_state.video_frame_proposals[frame_index]}
        preview_pairs = [(index, frame_pieces[piece.label.strip().lower()])
                         for index, piece in enumerate(proposals)
                         if st.session_state.get(f"piece_selected_{index}", True)
                         and piece.label.strip().lower() in frame_pieces]
    else:
        preview_pairs = [(index, piece) for index, piece in enumerate(proposals)
                         if st.session_state.get(f"piece_selected_{index}", True)]
    selected_for_preview = [piece for _, piece in preview_pairs]
    photo_column, pieces_column, poem_column = st.columns([1.08, 1.08, 1], gap="small")

    with photo_column:
        with st.container(key="video_photo_panel" if video_frames else "photo_panel", border=False):
            view_options = ("Analyzed frame", "Replay video", "Overlay replay") if video_frames and proposals else (
                "Analyzed frame", "Replay video")
            view = (st.radio("Preview", view_options,
                             horizontal=True, label_visibility="collapsed", key="video_view")
                    if video_frames else "Analyzed frame")
            if view == "Replay video":
                video_upload = st.session_state.get("video_upload")
                if video_upload is None:
                    st.write("Choose Change file and upload the clip again to enable replay.")
                else:
                    st.video(video_upload, format=st.session_state.video_format,
                             autoplay=False, width="stretch")
            elif view == "Overlay replay":
                video_upload = st.session_state.get("video_upload")
                if video_upload is None:
                    st.write("Choose Change file and upload the clip again to enable replay.")
                elif not selected_pieces:
                    st.write("Select at least one piece to show its overlay.")
                else:
                    labels = tuple(sorted({piece.label for piece in selected_pieces}))
                    if st.session_state.get("annotated_video_labels") != labels:
                        st.session_state.pop("annotated_video", None)
                    if "annotated_video" not in st.session_state:
                        with st.spinner("Rendering clothing overlays on the video…"):
                            try:
                                st.session_state.annotated_video = render_annotated_video(
                                    video_upload, Path(video_upload.name).suffix.lower(), video_frames,
                                    st.session_state.video_frame_proposals, set(labels))
                                st.session_state.annotated_video_labels = labels
                            except (ValueError, RuntimeError) as exc:
                                st.error(str(exc))
                    if "annotated_video" in st.session_state:
                        st.video(st.session_state.annotated_video, format="video/mp4", width="stretch")
                        st.write("Boxes use sampled detections; shapes are refreshed on rendered frames. The overlay replay has no audio.")
            else:
                preview = preview_selected_pieces(image, selected_for_preview)
                st.markdown(highlight_preview_html(preview, {
                    index: piece_highlight(image, piece) for index, piece in preview_pairs
                }), unsafe_allow_html=True)
                if not proposals and "piece_proposals" not in st.session_state:
                    st.write("Run detection to add clothing shapes and boxes to this image.")
                elif not proposals:
                    seg = st.session_state.get("segmentation")
                    st.write(seg.reason if seg is not None and seg.mask is None else
                             "No pieces had usable masks. Try a clearer photo or check the segmentation setup.")

    with pieces_column:
        st.subheader("Detected pieces")
        selected_items = []
        if proposals:
            with st.container(key="piece_list", border=False):
                for index, piece in enumerate(proposals):
                    with st.container(key=f"piece_row_{index}", border=False):
                        row = st.columns([.38, .68, 1.95, .55, 1.15], gap="small", vertical_alignment="center")
                        enabled = row[0].checkbox(f"Use {piece.label} {index + 1}", value=True,
                                                  key=f"piece_selected_{index}", label_visibility="collapsed")
                        source_image = (video_frames[st.session_state.video_piece_frames[index]].image
                                        if video_frames else image)
                        row[1].image(piece_thumbnail(source_image, piece.mask), width="stretch")
                        display_name = st.session_state.get(f"piece_name_{index}", piece.label)
                        default_color = color_name(piece.colors[0]) if piece.colors else "unknown"
                        display_color = st.session_state.get(f"piece_colors_{index}", default_color)
                        row[2].write(f"{display_name} · {display_color} · {piece.score:.0%}")
                        row[3].markdown(color_grid_html(piece.colors), unsafe_allow_html=True)
                        with row[4].popover("Edit"):
                            name = st.text_input("Piece name", piece.label, key=f"piece_name_{index}")
                            colors_text = st.text_input("Color name", default_color,
                                                        key=f"piece_colors_{index}")
                            outline_options = ["vertical", "broad", "balanced"]
                            default_outline = mask_outline(piece.mask)
                            outline = st.selectbox("Visible outline", outline_options,
                                                   index=outline_options.index(default_outline)
                                                   if default_outline in outline_options else 2,
                                                   key=f"piece_outline_{index}")
                        if enabled:
                            selected_items.append(PieceInput(piece, name,
                                (colors_text.strip(),) if colors_text.strip() else (), outline, source_image))
        elif "piece_proposals" in st.session_state:
            st.write("No pieces with usable masks were found.")
        else:
            st.write("Detected clothing pieces will appear here.")

    analysis = analysis_from_pieces(selected_items) if selected_items else None
    trace = map_analysis(analysis) if analysis else None
    basis = analysis.model_dump_json() if analysis else None
    if st.session_state.get("poem_basis") != basis:
        st.session_state.pop("poem", None)
        st.session_state.variation = 0

    with poem_column:
        st.subheader("Poem")
        if analysis is not None and analysis.overall.palette:
            st.markdown(outfit_palette_html(analysis.overall.palette), unsafe_allow_html=True)
        action = st.columns([1.2, 1], gap="small", vertical_alignment="center")
        generate = action[0].button("Generate poem", type="primary", disabled=analysis is None,
                                    width="stretch")
        with action[1].popover("Poem options"):
            tone = st.selectbox("Tone", ["lyrical", "minimal", "fragmentary"])
            length = st.slider("Lines", 8, 16, 12)
        if generate and analysis is not None and trace is not None:
            try:
                variation = st.session_state.get("variation", 0)
                poem = generate_poem(analysis, trace, length=length, tone=tone, variation=variation)
                st.session_state.analysis = analysis.model_dump()
                st.session_state.trace = trace.model_dump()
                st.session_state.poem = poem.model_dump()
                st.session_state.poem_basis = basis
                st.session_state.variation = variation + 1
            except Exception as exc:
                st.error(f"Poem generation failed: {exc}")

        with st.container(key="poem_body", border=False):
            if st.session_state.get("poem"):
                poem = Poem.model_validate(st.session_state.poem)
                st.subheader(poem.title)
                st.markdown('<div class="poem">' + "<br>".join(html.escape(line) for line in poem.lines) + '</div>',
                            unsafe_allow_html=True)
                with st.expander("Poem details and downloads"):
                    st.write("Motifs:", ", ".join(trace.dominant_motifs) or "none")
                    poem_text = poem.title + "\n\n" + "\n".join(poem.lines) + "\n"
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
