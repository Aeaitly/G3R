"""Download and verify the G3R pretrained model."""

import argparse
import hashlib
from pathlib import Path
import shutil
import urllib.request

RELEASE_URL = "https://github.com/Aeaitly/G3R/releases/download/v1.0.0"
MODELS = {
    "v1": ("g3r_v1.pt", "3c5c2ff02701f9be892d184ba9344a574dc336098cff01e3899e1594225b8b25"),
    "v2": ("g3r_v2.pt", "298b338513e15850f45c50a03566274f74ffe7c2d710bf669f705730b4038b71"),
}


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def download(model, output_dir):
    filename, checksum = MODELS[model]
    destination = output_dir / filename
    if destination.exists() and digest(destination) == checksum:
        print(f"Verified existing weights: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.download")
    url = f"{RELEASE_URL}/{filename}"
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if digest(temporary) != checksum:
        raise RuntimeError(f"SHA-256 mismatch for {temporary}; weights were not installed.")
    temporary.replace(destination)
    print(f"Verified weights: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="v2")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "pretrained",
    )
    args = parser.parse_args()
    for model in MODELS if args.model == "all" else [args.model]:
        download(model, args.output_dir)


if __name__ == "__main__":
    main()
