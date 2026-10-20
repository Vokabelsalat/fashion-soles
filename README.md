# Fashion Soles

Fashion Soles turns selected clothing pieces in a photograph into fictional associations and a poem. The local Streamlit workspace lets you upload a photo, detect pieces, select or edit them, and generate a poem. Use photographs of consenting adults. The poem does not make claims about the wearer.

## Setup

Python 3.10+ is supported. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-segmentation.txt
pip install -r requirements-fashionpedia.txt
python scripts/download_model.py
streamlit run app.py
```

The Fashionpedia detector downloads its third-party YOLOS checkpoint into the ignored `models/hf/` directory on first use. The MediaPipe model is stored in the ignored `models/` directory. These model downloads need a network connection once; detection and poem generation run locally afterward. No LLM API key or account is needed. If a model or dependency is unavailable, the app explains why detection cannot produce usable pieces.

## Workflow

1. Upload a JPG, PNG, or WEBP image up to 10 MB. The upload control disappears once the photo is loaded; **Change photo** returns to it.
2. Click **Detect pieces**. MediaPipe supplies coarse clothes and accessories masks, and Fashionpedia supplies labels and boxes. The app intersects them, removes near-duplicate detections, and keeps only pieces with usable masks.
3. Use the checkboxes to choose which pieces appear in the augmented photo and contribute to the poem. **Edit** lets you correct a piece name, color names, and a simple visible-outline description. The sampled color swatches remain visual references to the photograph.
4. Click **Generate poem**. The selected pieces become garment observations for the rules in `config/mappings.yaml`. A local template generator renders a poem from the selected names, colors, outlines, and mapped motifs. Generating again produces another variation. Changing a selection or edit clears the old poem.

The JSON download includes selected piece data, mapped rules, and the poem. The photo and masks are not included in that export.

## How the observations are used

The detector supplies the garment label and confidence. Up to three RGB colors are sampled from each provisional mask and converted to approximate names for the symbolic rules. The visible mask bounds supply a coarse `vertical`, `broad`, or `balanced` outline. This describes the mask on the page; it does not establish the garment's actual fit or construction. Color contrast between selected pieces can trigger a relationship rule. Material and condition remain unknown unless the user supplies them through another analysis path.

The poem generator is deterministic: the same selections, tone, line count, and variation produce the same poem. Its templates and symbolic associations are authored locally. Three tones and two template sets provide some variation, but repeated generations eventually reuse phrasing. The app makes no LLM inference calls and does not upload the photo or selected observations to a poem service.

The page aims to fit a desktop viewport with a bounded piece list. Smaller screens and long poems may require scrolling.

## Tests

```bash
pytest
```

## Limits and privacy

MediaPipe provides coarse categories rather than precise instance segmentation. A detector box can intersect the wrong clothes pixels, miss a piece, or assign an incorrect label. Its confidence is not a probability that the cropped mask is correct. Color names and outline descriptions are approximate. The mappings are authored fictional conventions, not personal profiling. The app does not save uploaded photos. There is no facial recognition or identity inference.
