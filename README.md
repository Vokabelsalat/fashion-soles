# Fashion Soles

Fashion Soles turns selected clothing pieces in a photo or short video into fictional associations and a poem. The local Streamlit workspace lets you upload media, detect pieces, select or edit them, and generate a poem. Use images of consenting adults. The poem does not make claims about the wearer.

## Setup

Python 3.10+ is supported. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-segmentation.txt
pip install -r requirements-fashionpedia.txt
python scripts/download_model.py
python scripts/download_language_data.py  # CMU dictionary and WordNet for poem forms and synonyms
streamlit run app.py
```

The Fashionpedia detector downloads its third-party YOLOS checkpoint (about 230 MB), and FashionCLIP its checkpoint (`patrickjohncyh/fashion-clip`, about 600 MB), into the ignored `models/hf/` directory on first use. The MediaPipe model is stored in the ignored `models/` directory. These model downloads need a network connection once; detection and poem generation run locally afterward. No LLM API key or account is needed. If a model or dependency is unavailable, the app explains why detection cannot produce usable pieces.

## Workflow

1. Upload a JPG, PNG, or WEBP photo up to 10 MB, or an MP4, MOV, or WEBM video up to 60 seconds and 1 GB. The upload control disappears once media is loaded; **Change file** returns to it. Videos are decoded locally into at most five evenly spaced sample frames. Use the frame selector in the header to review them, or choose **Replay video** above the image to play the original clip.
2. Click **Detect pieces**. MediaPipe supplies coarse clothes and accessories masks, and Fashionpedia supplies labels and boxes. The app intersects them, removes near-duplicate detections, and keeps only pieces with usable masks. Detections need a score of 0.45, but whole garments down to 0.25 are kept when no stronger piece already covers most of their pixels: the detector often splits one shirt over weak "shirt", "coat", and "top" boxes. For video, it analyzes each sampled frame and keeps the clearest detected instance of each label. The image preview shows boxes and masks for the currently selected frame; piece thumbnails come from their clearest frame.
3. Use the checkboxes to choose which pieces appear in the augmented photo and contribute to the poem. The palette above the poem ranks colors across selected garment pixels by frequency and shows a distinct accent when one has enough coverage. For video, choose **Overlay replay** to render a local MP4 with the selected boxes and shapes on the playing clip. This first render can take time; subsequent plays reuse it until the selection changes. **Edit** lets you correct a piece name, a color name, its pattern and material, and a simple visible-outline description. The piece swatch shows its sampled photo RGB, while the color name is interpreted for the poem and can be corrected.
4. Click **Generate poem**. The selected pieces become garment observations for the rules in `config/mappings.yaml`. A local template generator renders a poem from the selected names, colors, outlines, and mapped motifs. Generating again produces another variation. Changing a selection or edit clears the old poem.

The JSON download includes selected piece data, mapped rules, the poem, and video sample times when applicable. Uploaded media and masks are not included in that export.

## How the observations are used

The detector supplies the garment label and confidence. One dominant RGB color is sampled from each provisional mask and converted to an approximate name for the symbolic rules. The sampler first separates a few color clusters so shadows do not pull the single swatch toward their average. Separately, the outfit palette samples pixels inside all selected masks, groups similar shades, and ranks the primary and secondary colors by pixel frequency. It includes an accent when another color is substantially different from both and covers at least 1.5% of the sampled pixels. The palette is included in the analysis JSON and supplies a color line in every generated poem. Neutral colors are interpreted generously in dim or warm lighting; you can correct an individual piece's color name in **Edit**. The visible mask bounds supply a coarse `vertical`, `broad`, or `balanced` outline. This describes the mask on the page; it does not establish the garment's actual fit or construction. Color contrast between selected pieces can trigger a relationship rule. FashionCLIP, a CLIP model fine-tuned on fashion product photos, compares each piece with text prompts for patterns (solid, striped, plaid, houndstooth, polka dot, floral, paisley, animal print, camouflage, tie-dye, graphic, geometric) and materials (denim, leather, knit, wool, silk, and others; fewer, likelier ones for shoes and bags). It sees the piece's bounds with a little context, padded to a square, with the other detected pieces greyed out so a neighbour's print is not read as this piece's. Answers below a confidence floor are dropped: for an unsure pattern, the edge, orientation, and spectrum measures in `texture.py` may still name stripes, checks, or dots, which they measure well; otherwise the piece reads as plain, textured, or busy. An unsure material stays unknown. Material is the weaker of the two on real photos and can be corrected in **Edit**. If the model is unavailable, `texture.py` alone describes the pieces. Condition remains unknown.

The poem generator is deterministic: the same selections, tone, line count, and variation produce the same poem. Its templates and symbolic associations are authored locally. Three tones and two template sets provide some variation, but repeated generations eventually reuse phrasing. The app makes no LLM inference calls and does not upload the photo or selected observations to a poem service.

On desktop, the selected photo or video frame, detected pieces, and poem sit in three columns. The image uses the available viewport height; long piece lists and poems scroll within their columns. Smaller screens may require page scrolling.

## Tests

```bash
pytest
python scripts/evaluate_photos.py
```

`tests/fixtures/photos/labels.csv` lists the expected pieces, patterns, and materials for a few real photos. Rows with status `pass` are checked by `pytest`; `known miss` rows document current failures. Add a photo to that folder with one row per piece to extend the set. The FashionCLIP tests are skipped when its weights are not downloaded; other tests replace the model with a stub.

## Limits and privacy

The video analysis still detects objects in at most five sampled frames, rather than tracking them. Overlay replay interpolates boxes between neighboring samples and reruns MediaPipe for a coarse shape on each rendered frame. It renders at up to 6 frames per second and 960 pixels on the longest side; moving boxes may drift or briefly disappear. The annotated MP4 has no audio, so use **Replay video** for the original sound and frame rate. Rendering requires an H.264 encoder in OpenCV and may take time or use substantial local memory. A piece visible only between samples may be missed; separate pieces with the same label may be combined. Streamlit is configured for uploads up to 1 GB, and the video decoder streams the upload through a temporary file. The source video remains in the local Streamlit session for replay and is not written to the analysis export. MP4, MOV, and WEBM decoding depends on the installed OpenCV codecs; replay also depends on the browser supporting the video codec. MediaPipe provides coarse categories rather than precise instance segmentation. A detector box can intersect the wrong clothes pixels, miss a piece, or assign an incorrect label. Its confidence is not a probability that the cropped mask is correct. Color names and outline descriptions are approximate. The mappings are authored fictional conventions, not personal profiling. The app does not save uploaded photos. There is no facial recognition or identity inference.
