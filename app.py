from __future__ import annotations

import json
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from fashion_soles.mapping import map_analysis
from fashion_soles.pipeline import clear_piece_selection, demo_analysis, export_trace, image_fingerprint, reset_for_new_image
from fashion_soles.presentation import color_grid_html
from fashion_soles.poetry import demo_poem, generate_poem
from fashion_soles.preprocessing import prepare_image
from fashion_soles.schemas import OutfitAnalysis
from fashion_soles.segmentation import overlay, segment_clothes
from fashion_soles.fashion_detector import propose_pieces, preview_selected_pieces
from fashion_soles.vision import analyze_image

load_dotenv()
DEMO = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
st.set_page_config(page_title="Fashion Soles", page_icon="✦", layout="wide")
st.markdown("""<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#f5f2eb; color:#292720;}
.block-container {max-width:1120px; padding-top:2.8rem; padding-bottom:5rem;}
h1,h2,h3 {font-family:Georgia,serif !important; font-weight:400 !important; color:#292720 !important;}
h1 {font-size:3.5rem !important; letter-spacing:.04em; margin-bottom:0 !important;}
h2 {font-size:2rem !important; margin-top:2.5rem !important;}
p, label, .stMarkdown {color:#292720;}
div.stButton button, div.stFormSubmitButton button, div.stDownloadButton button,
[data-testid="stFileUploader"] button {
  background:#f5f2eb !important; border:1px solid #292720 !important;
  color:#292720 !important; border-radius:2px !important;
}
div.stButton button:hover, div.stButton button:focus,
div.stFormSubmitButton button:hover, div.stFormSubmitButton button:focus,
div.stDownloadButton button:hover, div.stDownloadButton button:focus,
[data-testid="stFileUploader"] button:hover, [data-testid="stFileUploader"] button:focus {
  background:#f5f2eb !important; border-color:#292720 !important; color:#292720 !important;
  text-decoration:underline;
}
div.stButton button[kind="primary"] {font-weight:700;}
div[data-testid="stFileUploader"] {border-radius:2px;}
.poem {font-family:Georgia,serif; font-size:1.24rem; line-height:1.72; white-space:pre-line;}
.subtitle {font-family:Georgia,serif; font-size:1.25rem; margin-top:-.4rem;}
.piece-color-grid {display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:.45rem; width:100%; max-width:14rem;}
.piece-swatch {height:2rem; border:1px solid #292720; border-radius:2px;}
</style>""", unsafe_allow_html=True)

st.title("Fashion Soles")
st.markdown('<p class="subtitle">Every outfit is an unwritten story.</p>', unsafe_allow_html=True)
st.write("An experiment in fictional associations drawn from visible clothing. Photograph only with permission; use an image of one consenting adult. The poem makes no claims about the wearer.")
if DEMO:
    st.info("Demo data: no image analysis performed. The committed fictional outfit is used for observations and poetry.")
else:
    st.write("Clicking Analyze photo sends the processed image to OpenAI. Uploads stay in this session and are not saved by this app.")

st.header("1. Photograph")
upload = st.file_uploader("Choose a JPG, PNG, or WEBP photo (up to 10 MB)", type=["jpg", "jpeg", "png", "webp"],
                          key=f"upload_{st.session_state.get('upload_epoch', 0)}")
raw = upload.getvalue() if upload else None
fingerprint = image_fingerprint(raw) if raw else None
reset_for_new_image(st.session_state, fingerprint)
image = None
if raw:
    try:
        image = prepare_image(raw)
        st.image(image, caption="Processed original · EXIF removed", width="stretch")
    except ValueError as exc:
        st.error(str(exc))
        image = None
if st.button("Clear and reset"):
    epoch = st.session_state.get("upload_epoch", 0) + 1
    st.session_state.clear()
    st.session_state.upload_epoch = epoch
    st.rerun()
if not image and not DEMO:
    st.info("Choose a photo with one clearly visible outfit. If the subject is cropped or several people are present, the analysis may be ambiguous.")

can_analyze = DEMO or image is not None
if st.button("Analyze photo" if not DEMO else "Load demo analysis", type="primary", disabled=not can_analyze):
    try:
        analysis = demo_analysis() if DEMO else analyze_image(image)
        st.session_state.analysis = analysis.model_dump()
        st.session_state.trace = map_analysis(analysis).model_dump()
        st.session_state.poem = None
        st.session_state.variation = 0
        st.session_state.segmentation = segment_clothes(image) if image else None
        st.session_state.pop("piece_proposals", None)
        clear_piece_selection(st.session_state)
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

if not DEMO and image is not None and not os.getenv("OPENAI_API_KEY"):
    st.warning("No OpenAI API key is configured. Add one to .env or restart with DEMO_MODE=true.")

if image is not None:
    st.header("2. Segmentation")
    seg = st.session_state.get("segmentation")
    if seg is None:
        st.write("Run analysis to try the local clothes mask.")
    elif seg.mask is None:
        st.warning(f"{seg.status}. {seg.reason}")
    else:
        st.write(f"Coarse clothes mask · {seg.clothes_percentage}% of pixels. This does not separate individual garments or reliably include accessories.")
        opacity = st.slider("Overlay opacity", 0.0, 1.0, .45, .05)
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Original", width="stretch")
        with right:
            st.image(overlay(image, seg.mask, opacity), caption="Clothes overlay", width="stretch")
        if st.checkbox("Show mask only"):
            st.image(seg.mask, caption="Clothes category only", width="stretch")
    st.subheader("Garment proposals")
    st.write("A Fashionpedia-trained detector finds garment boxes. Each provisional piece is the box intersected with MediaPipe's clothes or accessories category. Near-duplicate detections with the same label are filtered. Different garment boxes can still claim the same pixels.")
    threshold = st.slider("Detection confidence", 0.1, 0.9, 0.45, 0.05)
    if st.button("Detect garment pieces"):
        with st.spinner("Detecting garment boxes and building provisional masks. The first run downloads the local model."):
            try:
                if seg is None:
                    seg = segment_clothes(image)
                    st.session_state.segmentation = seg
                st.session_state.piece_proposals = propose_pieces(
                    image, seg.mask, seg.accessory_mask, threshold)
                clear_piece_selection(st.session_state)
                st.rerun()
            except Exception as exc:
                st.error(f"Garment detection unavailable: {type(exc).__name__}: {exc}")
    proposals = st.session_state.get("piece_proposals")
    if proposals is not None:
        if not proposals:
            st.info("No garment boxes met this confidence threshold. Try a lower threshold or a clearer full-outfit photo.")
        else:
            st.write("Select pieces to show in the provisional intersection view. All pieces are selected when detection finishes.")
            headers = st.columns([.65, 1.1, 2.2, 1.2, 2.3], vertical_alignment="center")
            for column, title in zip(headers, ("Show", "Mask", "Label", "Confidence", "Colors")):
                column.write(title)
            selected = []
            for index, piece in enumerate(proposals):
                columns = st.columns([.65, 1.1, 2.2, 1.2, 2.3], vertical_alignment="center")
                with columns[0]:
                    enabled = st.checkbox(f"Show {piece.label} {index + 1}", value=True,
                                          key=f"piece_selected_{index}", label_visibility="collapsed")
                with columns[1]:
                    if piece.mask is not None and piece.mask.getbbox():
                        st.image(piece.mask.crop(piece.mask.getbbox()), width=64)
                    else:
                        st.write("Unavailable")
                columns[2].write(piece.label)
                columns[3].write(f"{piece.score:.3f}")
                columns[4].markdown(color_grid_html(piece.colors), unsafe_allow_html=True)
                if enabled:
                    selected.append(piece)
                st.divider()
            st.write("Colors come from provisional mask pixels. Missing masks have no sampled colors.")
            st.subheader("Provisional intersections")
            if not selected:
                st.write("Select a piece above to show its intersection or detection box.")
            st.image(preview_selected_pieces(image, selected), width="stretch")

if "analysis" in st.session_state:
    analysis = OutfitAnalysis.model_validate(st.session_state.analysis)
    st.header("3. Observation")
    if analysis.image_suitability != "suitable":
        st.warning(f"Image suitability: {analysis.image_suitability}. Consider a clearer photograph of one outfit.")
    for warning in analysis.warnings:
        st.warning(warning)
    st.write("Model observations can be uncertain. Material stays unknown when it cannot be identified from the photograph.")
    st.write("Overall colours:", ", ".join(analysis.overall.dominant_colors) or "unknown")
    st.write("Overall silhouette:", ", ".join(analysis.overall.silhouette) or "unknown")
    with st.form("corrections"):
        updated = analysis.model_dump()
        for i, garment in enumerate(analysis.garments):
            st.subheader(f"{garment.name} · {garment.id}")
            c1, c2 = st.columns(2)
            with c1:
                updated["garments"][i]["category"] = st.text_input("Category", garment.category, key=f"cat_{garment.id}")
                colors = st.text_input("Colours (comma separated)", ", ".join(garment.colors), key=f"colors_{garment.id}")
                updated["garments"][i]["colors"] = [v.strip() for v in colors.split(",") if v.strip()]
                updated["garments"][i]["material"] = st.text_input("Material", garment.material, key=f"mat_{garment.id}")
            with c2:
                silhouette = st.text_input("Silhouette (comma separated)", ", ".join(garment.silhouette), key=f"sil_{garment.id}")
                condition = st.text_input("Visible condition (comma separated)", ", ".join(garment.condition), key=f"cond_{garment.id}")
                updated["garments"][i]["silhouette"] = [v.strip() for v in silhouette.split(",") if v.strip()]
                updated["garments"][i]["condition"] = [v.strip() for v in condition.split(",") if v.strip()]
            st.write("Texture:", ", ".join(garment.texture) or "unknown", " · Confidence:", garment.confidence if garment.confidence is not None else "unknown")
        applied = st.form_submit_button("Apply corrections")
    if applied:
        try:
            corrected = OutfitAnalysis.model_validate(updated)
            st.session_state.analysis = corrected.model_dump()
            st.session_state.trace = map_analysis(corrected).model_dump()
            st.session_state.poem = None
            st.session_state.variation = 0
            st.success("Corrections applied. Mapping updated without another image analysis call.")
            st.rerun()
        except ValidationError as exc:
            st.error(f"Invalid correction: {exc}")

    from fashion_soles.schemas import MappingTrace, Poem
    trace = MappingTrace.model_validate(st.session_state.trace)
    st.header("4. Associations")
    st.write("These are editable creative conventions in config/mappings.yaml, not claims about the wearer.")
    if trace.matched_rules:
        for match in trace.matched_rules:
            st.markdown(f"**{match.rule_id}** · {match.source} → {', '.join(match.motifs)}. {match.rationale}")
    else:
        st.write("No rules matched the visible observations.")
    st.write("Dominant motifs:", ", ".join(trace.dominant_motifs) or "none")
    for tension in trace.tensions:
        st.write(f"{tension.rule_id}: {tension.description} ({' / '.join(tension.between)})")

    st.header("5. Poem")
    col1, col2 = st.columns(2)
    with col1:
        tone = st.selectbox("Tone", ["lyrical", "minimal", "fragmentary"])
    with col2:
        length = st.slider("Lines", 8, 16, 12)
    generate = st.button("Generate poem", type="primary")
    regenerate = st.button("Regenerate poem", disabled=not st.session_state.get("poem"))
    if generate or regenerate:
        try:
            variation = st.session_state.get("variation", 0) + (1 if regenerate else 0)
            st.session_state.variation = variation
            poem = (demo_poem(analysis, trace, length=length, tone=tone, variation=variation) if DEMO
                    else generate_poem(analysis, trace, length=length, tone=tone, variation=variation))
            st.session_state.poem = poem.model_dump()
            st.rerun()
        except Exception as exc:
            st.error(f"Poem generation failed: {exc}")
    if st.session_state.get("poem"):
        poem = Poem.model_validate(st.session_state.poem)
        st.subheader(poem.title)
        st.markdown('<div class="poem">' + "<br>".join(__import__("html").escape(line) for line in poem.lines) + '</div>', unsafe_allow_html=True)
        st.write("Drawn from rules:", ", ".join(poem.used_rule_ids) or "none")
        poem_text = poem.title + "\n\n" + "\n".join(poem.lines) + "\n"
        st.code(poem_text, language=None)
        st.download_button("Download poem (.txt)", poem_text, file_name="fashion-soles-poem.txt", mime="text/plain")
    seg = st.session_state.get("segmentation")
    st.download_button("Download full JSON trace", export_trace(analysis, trace,
        Poem.model_validate(st.session_state.poem) if st.session_state.get("poem") else None,
        mode="demo" if DEMO else "api", segmentation_status=seg.status if seg else "not run",
        pieces=st.session_state.get("piece_proposals")),
        file_name="fashion-soles-trace.json", mime="application/json")
