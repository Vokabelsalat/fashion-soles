from __future__ import annotations

import html
import os

import streamlit as st
from dotenv import load_dotenv

from fashion_soles.fashion_detector import available_pieces, piece_thumbnail, preview_selected_pieces, propose_pieces
from fashion_soles.from_pieces import PieceInput, analysis_from_pieces, color_name, mask_outline
from fashion_soles.mapping import map_analysis
from fashion_soles.pipeline import clear_piece_controls, export_trace, image_fingerprint, reset_for_new_image
from fashion_soles.poetry import demo_poem, generate_poem
from fashion_soles.preprocessing import prepare_image
from fashion_soles.presentation import color_grid_html
from fashion_soles.schemas import MappingTrace, OutfitAnalysis, Poem
from fashion_soles.segmentation import segment_clothes

load_dotenv()
DEMO = os.getenv("DEMO_MODE", "false").lower() in {"1", "true", "yes"}
HAS_TEXT_API = bool(os.getenv("OPENAI_API_KEY")) and not DEMO
st.set_page_config(page_title="Fashion Soles", page_icon="✦", layout="wide")
st.markdown("""<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {background:#f5f2eb; color:#292720;}
.block-container {max-width:1280px; padding-top:1.1rem; padding-bottom:1rem;}
h1,h2,h3 {font-family:Georgia,serif !important; font-weight:400 !important; color:#292720 !important;}
h1 {font-size:2.4rem !important; margin-bottom:.2rem !important;}
h2,h3 {font-size:1.45rem !important; margin-top:.6rem !important;}
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
.piece-color-grid {display:grid; grid-auto-flow:column; grid-auto-columns:1fr; gap:.3rem; width:100%;}
.piece-swatch {height:1.5rem; border:1px solid #292720; border-radius:2px;}
.poem {font-family:Georgia,serif; font-size:1.08rem; line-height:1.42; white-space:pre-line;}
hr {margin:.35rem 0 !important;}
</style>""", unsafe_allow_html=True)

st.title("Fashion Soles")

image = st.session_state.get("photo")
if image is None:
    st.write("Upload a photograph of one consenting adult. The local models find clothing pieces; the poem is a fictional reading of what is visible.")
    upload = st.file_uploader("Choose a JPG, PNG, or WEBP photo (up to 10 MB)",
                              type=["jpg", "jpeg", "png", "webp"],
                              key=f"upload_{st.session_state.get('upload_epoch', 0)}")
    if upload is not None:
        try:
            raw = upload.getvalue()
            prepared = prepare_image(raw)
            reset_for_new_image(st.session_state, image_fingerprint(raw))
            st.session_state.photo = prepared
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
else:
    toolbar = st.columns([1, 1, 4], vertical_alignment="center")
    if toolbar[0].button("Change photo", use_container_width=True):
        epoch = st.session_state.get("upload_epoch", 0) + 1
        st.session_state.clear()
        st.session_state.upload_epoch = epoch
        st.rerun()
    detect = toolbar[1].button("Detect pieces", type="primary", use_container_width=True)
    if detect:
        with st.spinner("Finding clothing pieces with the local models…"):
            try:
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
    selected_for_preview = [piece for index, piece in enumerate(proposals)
                            if st.session_state.get(f"piece_selected_{index}", True)]
    left, right = st.columns([1.18, 1], gap="medium")
    with left:
        preview = preview_selected_pieces(image, selected_for_preview)
        preview.thumbnail((720, 540))
        st.image(preview, width=preview.width)
        if proposals:
            st.write(f"{len(selected_for_preview)} of {len(proposals)} detected pieces shown in the preview.")
        elif "piece_proposals" not in st.session_state:
            st.write("Run detection to add clothing shapes and boxes to this image.")
        else:
            seg = st.session_state.get("segmentation")
            st.write(seg.reason if seg is not None and seg.mask is None else
                     "No pieces had usable masks. Try a clearer photo or check the segmentation setup.")

    with right:
        st.subheader("Detected pieces")
        selected_items = []
        if proposals:
            with st.container(height=345, border=False):
                for index, piece in enumerate(proposals):
                    row = st.columns([.44, .8, 2.25, 1.1, .76], vertical_alignment="center")
                    enabled = row[0].checkbox(f"Use {piece.label} {index + 1}", value=True,
                                              key=f"piece_selected_{index}", label_visibility="collapsed")
                    row[1].image(piece_thumbnail(image, piece.mask), width=64)
                    display_name = st.session_state.get(f"piece_name_{index}", piece.label)
                    row[2].write(f"{display_name} · {piece.score:.0%}")
                    row[3].markdown(color_grid_html(piece.colors), unsafe_allow_html=True)
                    default_colors = ", ".join(dict.fromkeys(color_name(value) for value in piece.colors))
                    with row[4].popover("Edit"):
                        name = st.text_input("Piece name", piece.label, key=f"piece_name_{index}")
                        colors_text = st.text_input("Color names, separated by commas", default_colors,
                                                    key=f"piece_colors_{index}")
                        outline_options = ["vertical", "broad", "balanced"]
                        default_outline = mask_outline(piece.mask)
                        outline = st.selectbox("Visible outline", outline_options,
                                               index=outline_options.index(default_outline)
                                               if default_outline in outline_options else 2,
                                               key=f"piece_outline_{index}")
                    if enabled:
                        selected_items.append(PieceInput(piece, name,
                            tuple(color.strip() for color in colors_text.split(",") if color.strip()), outline))
                    st.divider()
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

        action = st.columns([1.2, 1], vertical_alignment="center")
        generate = action[0].button("Generate poem", type="primary", disabled=analysis is None,
                                    use_container_width=True)
        with action[1].popover("Poem options"):
            tone = st.selectbox("Tone", ["lyrical", "minimal", "fragmentary"])
            length = st.slider("Lines", 8, 16, 12)
        if generate and analysis is not None and trace is not None:
            try:
                variation = st.session_state.get("variation", 0)
                poem = (generate_poem(analysis, trace, length=length, tone=tone, variation=variation)
                        if HAS_TEXT_API else demo_poem(analysis, trace, length=length,
                                                       tone=tone, variation=variation))
                st.session_state.analysis = analysis.model_dump()
                st.session_state.trace = trace.model_dump()
                st.session_state.poem = poem.model_dump()
                st.session_state.poem_basis = basis
                st.session_state.variation = variation + 1
            except Exception as exc:
                st.error(f"Poem generation failed: {exc}")

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
                    analysis, trace, poem, mode="text_api" if HAS_TEXT_API else "template",
                    segmentation_status=seg.status if seg else "not run", pieces=selected_for_preview),
                    file_name="fashion-soles-trace.json", mime="application/json")
        elif not HAS_TEXT_API:
            st.write("Poems use a local template preview. Set OPENAI_API_KEY for original generated poems.")
