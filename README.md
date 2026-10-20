# Fashion Soles

Fashion Soles is a local, one-page experiment that turns observed outfit details into fictional associations and an 8–16-line poem. The poem is an invented reading of clothing, not a claim about the person wearing it. Use only photographs of consenting adults.

## Start

Python 3.11+ is recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

For a fully offline demo, set `DEMO_MODE=true` in `.env` or run:

```bash
DEMO_MODE=true streamlit run app.py
```

Click **Load demo analysis** even without uploading a photo. The demo uses `tests/fixtures/example_analysis.json`, and its poems are explicitly labelled templates. If you upload a photo in demo mode, the fixture still supplies the observations; no image recognition is claimed. A local segmentation model, if installed, can still make a real mask.

## API configuration

Edit `.env`:

```text
OPENAI_API_KEY=your_key_here
OPENAI_VISION_MODEL=gpt-4.1-mini
OPENAI_TEXT_MODEL=gpt-4.1-mini
DEMO_MODE=false
SEGMENTATION_MODEL_PATH=models/selfie_multiclass_256x256.tflite
```

Clicking **Analyze photo** sends a resized, EXIF-stripped JPEG to OpenAI's Responses API for structured outfit observations. That is the only image API call. **Apply corrections** recomputes mappings locally, and **Generate poem**/**Regenerate poem** send only the corrected analysis and mapping trace to a separate text call. Responses use `store=False`. The app does not save uploads or API keys.

## Local clothing mask

MediaPipe is optional because its wheels vary by platform. Install it and the official model:

```bash
pip install -r requirements-segmentation.txt
python scripts/download_model.py
```

The model is Google's [SelfieMulticlass 256×256](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter#multi-class_selfie_segmentation_model). It marks category `4` as clothes. The mask is coarse: it cannot separate a coat from trousers and may omit accessories. If the model file or package is absent, or inference fails, the app reports **Segmentation unavailable** and continues. The model is downloaded to the ignored `models/` directory; the script uses the official link from Google's model page.

## Provisional garment pieces

For automatic garment boxes, install the optional Fashionpedia detector dependencies:

```bash
pip install -r requirements-fashionpedia.txt
```

Upload a photo and click **Detect garment pieces**. On first use, the app downloads a [third-party YOLOS checkpoint fine-tuned on Fashionpedia](https://huggingface.co/valentinafevu/yolos-fashionpedia) into the ignored `models/hf/` directory. The photograph stays local for this stage. The detector supplies labels and boxes; the app intersects each box with MediaPipe's clothes category, or its `others` category for accessories. Detections without usable intersection masks are filtered out; if MediaPipe is unavailable, the list is empty and the app explains why. Same-label proposals with strongly overlapping, similarly sized regions are filtered, favouring confidence and area; separate items remain. Each resulting piece mask is explicitly provisional: overlapping boxes can contain the same clothes pixels, and a missed coarse-mask region cannot be recovered by intersection. The detector does not replace the editable vision observations or affect the poem automatically.

The detected-pieces list shows a padded thumbnail made from each mask and the original photo pixels, detector label, confidence score, and up to three dominant colors as swatches in one row. Pixels outside the mask are transparent. Each piece is selected by default; its checkbox controls whether its provisional mask and detector box appear in the combined preview below. Re-running detection or choosing a new image resets the selection. Colors are sampled only from pixels inside each provisional mask, after duplicate removal. A nearly uniform piece can have fewer than three distinct colors. The JSON trace includes the RGB hex values but omits mask image data.

## Inspect and change the work

The five sections show the photo, mask, observations, triggered rule IDs and sources, and poem. Edit category, colours, material, silhouette, or visible condition, then click **Apply corrections**. Rules and combinations live in `config/mappings.yaml`; edit that file and restart the app. The poem instructions live in `fashion_soles/poetry.py`. Download the poem as text or the full pipeline trace as JSON. The poem is also shown in a copyable text block.

Run tests with:

```bash
pytest
```

## Limits and privacy

Vision can misread garments, materials, and the number of people. Unknown material stays unknown unless visually supportable. A full outfit on one person works best; cropped or multiple-person images receive warnings, and the user should choose a clearer photo. The mask is a diagnostic overlay, not garment recognition. Symbolic motifs are authored artistic conventions, not personal profiling. OpenAI calls require network access and may incur cost. Demo poems are deliberately template-based and are not presented as AI writing. No facial recognition or identity inference is performed.

This first version has no video or real-time AR. Future work could improve garment-level segmentation, evaluate poem quality against a consented image set, and add live camera processing.
