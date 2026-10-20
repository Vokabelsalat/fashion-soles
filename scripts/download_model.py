"""Download the official MediaPipe SelfieMulticlass model."""
from pathlib import Path
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
DEST = Path(__file__).resolve().parents[1] / "models" / "selfie_multiclass_256x256.tflite"


def main():
    DEST.parent.mkdir(exist_ok=True)
    with urlopen(URL, timeout=30) as response:
        data = response.read(30_000_001)
    if not data or len(data) > 30_000_000:
        raise RuntimeError("Unexpected model download size")
    DEST.write_bytes(data)
    print(f"Downloaded official model to {DEST} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
