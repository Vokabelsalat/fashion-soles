"""Download the CMU Pronouncing Dictionary and WordNet for poem forms and synonyms.

Both are stored under models/ and used offline afterwards:
- models/cmudict: the CMU dictionary from the CMU Sphinx project (BSD license);
- models/nltk_data: WordNet through NLTK (WordNet license, permissive).
"""
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "models"
CMUDICT = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/"


def main():
    target = ROOT / "cmudict"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("cmudict.dict", "LICENSE"):
        with urlopen(CMUDICT + name, timeout=60) as response:
            data = response.read(10_000_001)
        if not data or len(data) > 10_000_000:
            raise RuntimeError(f"Unexpected download size for {name}")
        (target / name).write_bytes(data)
    print(f"Downloaded the CMU dictionary to {target}")

    import nltk
    if not nltk.download("wordnet", download_dir=str(ROOT / "nltk_data"), quiet=True):
        raise RuntimeError("WordNet download failed")
    print(f"Downloaded WordNet to {ROOT / 'nltk_data'}")


if __name__ == "__main__":
    main()
