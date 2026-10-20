# Fashion Soles

Fashion Soles turns selected clothing pieces in a photograph into fictional associations and a poem. The interface is a compact local Streamlit workspace: upload a photo, detect pieces, select or edit them, and generate a poem. Use photographs of consenting adults. The poem does not make claims about the wearer.

## Setup

Python 3.10+ is supported. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-segmentation.txt
pip install -r requirements-fashionpedia.txt
python scripts/download_model.py
cp .env.example .env
streamlit run app.py
```

The Fashionpedia detector downloads its third-party YOLOS checkpoint into the ignored `models/hf/` directory on first use. The MediaPipe model is stored in the ignored `models/` directory. If either model or its dependency is unavailable, the app explains why detection cannot produce usable pieces. The generated-poem path needs an `OPENAI_API_KEY` in `.env`; without one, the app provides a clearly titled local **Demo poem** template so the complete interaction can be tried offline. Set `DEMO_MODE=true` to force that template even when a key is present.

## Workflow

1. Upload a JPG, PNG, or WEBP image up to 10 MB. The upload control disappears once the photo is loaded; **Change photo** returns to it.
2. Click **Detect pieces**. MediaPipe supplies coarse clothes and accessories masks, and the Fashionpedia detector supplies labels and boxes. The app intersects them, removes near-duplicate detections, and keeps only pieces with usable masks.
3. Use the checkboxes to choose which pieces appear in the augmented photo and contribute to the poem. **Edit** lets you correct a piece name, color names, and a simple visible-outline description. The sampled color swatches remain visual references to the photograph.
4. Click **Generate poem**. The selected pieces become garment observations for the rules in `config/mappings.yaml`. A text-only OpenAI call produces an original poem when a key is configured; the local template is labelled as a demo. Generating again produces another variation. Changing a selection or edit clears the old poem.

The JSON download includes selected piece data, mapped rules, and the poem. The photo and masks are not included in that export.

## How the observations are used

The detector supplies the garment label and confidence. Up to three RGB colors are sampled from each provisional mask and converted to approximate names for the symbolic rules. The visible mask bounds supply a coarse `vertical`, `broad`, or `balanced` outline. This describes the mask on the page; it does not establish the garment's actual fit or construction. Color contrast between selected pieces can trigger a relationship rule. Material and condition remain unknown unless the user supplies them through another analysis path. The existing image-vision module remains in the codebase but is not called by this interface; uploaded photos stay local during detection, and only selected structured observations go to the poem API.

The page aims to fit a desktop viewport with a bounded piece list. Smaller screens and long poems may require scrolling.

## Tests

```bash
pytest
```

## Limits and privacy

MediaPipe provides coarse categories rather than precise instance segmentation. A detector box can intersect the wrong clothes pixels, miss a piece, or assign an incorrect label. Its confidence is not a probability that the cropped mask is correct. Color names and outline descriptions are approximate. The mappings are authored fictional conventions, not personal profiling. The local demo uses templates and is not presented as AI writing. OpenAI text calls require network access and may incur cost; calls use `store=False`. The app does not save uploaded photos or API keys. There is no facial recognition or identity inference.
