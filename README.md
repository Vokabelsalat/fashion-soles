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
streamlit run app.py
```

The Fashionpedia detector downloads its third-party YOLOS checkpoint into the ignored `models/hf/` directory on first use. The MediaPipe model is stored in the ignored `models/` directory. These model downloads need a network connection once; detection and poem generation run locally afterward. No LLM API key or account is needed. If a model or dependency is unavailable, the app explains why detection cannot produce usable pieces.

## Workflow

1. Upload a JPG, PNG, or WEBP photo up to 10 MB, or an MP4, MOV, or WEBM video up to 60 seconds and 1 GB. The upload control disappears once media is loaded; **Change file** returns to it. Videos are decoded locally into at most five evenly spaced sample frames. Use the frame selector in the header to review them, or choose **Replay video** above the image to play the original clip.
2. Click **Detect pieces**. MediaPipe supplies coarse clothes and accessories masks, and Fashionpedia supplies labels and boxes. The app intersects them, removes near-duplicate detections, and keeps only pieces with usable masks. For video, it analyzes each sampled frame and keeps the clearest detected instance of each label. The image preview shows boxes and masks for the currently selected frame; piece thumbnails come from their clearest frame.
3. Use the checkboxes to choose which pieces appear in the augmented photo and contribute to the poem. **Edit** lets you correct a piece name, a color name, and a simple visible-outline description. The swatch shows the sampled photo RGB, while the color name is interpreted for the poem and can be corrected.
4. Click **Generate poem**. The selected pieces become garment observations for the rules in `config/mappings.yaml`. A local template generator renders a poem from the selected names, colors, outlines, and mapped motifs. Generating again produces another variation. Changing a selection or edit clears the old poem.

The JSON download includes selected piece data, mapped rules, the poem, and video sample times when applicable. Uploaded media and masks are not included in that export.

## How the observations are used

The detector supplies the garment label and confidence. One dominant RGB color is sampled from each provisional mask and converted to an approximate name for the symbolic rules. The sampler first separates a few color clusters so shadows do not pull the single swatch toward their average. Neutral colors are interpreted generously in dim or warm lighting; you can correct the name in **Edit**. The visible mask bounds supply a coarse `vertical`, `broad`, or `balanced` outline. This describes the mask on the page; it does not establish the garment's actual fit or construction. Color contrast between selected pieces can trigger a relationship rule. Material and condition remain unknown unless the user supplies them through another analysis path.

The poem generator is deterministic: the same selections, tone, line count, and variation produce the same poem. Its templates and symbolic associations are authored locally. Three tones and two template sets provide some variation, but repeated generations eventually reuse phrasing. The app makes no LLM inference calls and does not upload the photo or selected observations to a poem service.

On desktop, the selected photo or video frame, detected pieces, and poem sit in three columns. The image uses the available viewport height; long piece lists and poems scroll within their columns. Smaller screens may require page scrolling.

## Tests

```bash
pytest
```

## Limits and privacy

The video view is a sampled-frame analysis, not continuous playback or object tracking. A piece visible only between samples may be missed; separate pieces with the same label may be combined. Streamlit is configured for uploads up to 1 GB, and the video decoder streams the upload through a temporary file. The source video remains in the local Streamlit session for replay and is not written to the analysis export. Detection can take longer because it runs on several frames. MP4, MOV, and WEBM decoding depends on the installed OpenCV codecs; replay also depends on the browser supporting the video codec. Replaying a very large upload may use substantial local memory. MediaPipe provides coarse categories rather than precise instance segmentation. A detector box can intersect the wrong clothes pixels, miss a piece, or assign an incorrect label. Its confidence is not a probability that the cropped mask is correct. Color names and outline descriptions are approximate. The mappings are authored fictional conventions, not personal profiling. The app does not save uploaded photos. There is no facial recognition or identity inference.
