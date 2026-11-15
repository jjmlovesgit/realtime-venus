#!/usr/bin/env python3
"""Download the small local English model used for microphone diagnostics."""

from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import ZipFile
import shutil


NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
ROOT = Path(__file__).resolve().parent / "models"
MODEL = ROOT / NAME


def main():
    if (MODEL / "am" / "final.mdl").is_file():
        print(f"Ready: {MODEL}")
        return
    ROOT.mkdir(exist_ok=True)
    archive = ROOT / f"{NAME}.zip.download"
    try:
        print(f"Downloading {URL} ...", flush=True)
        with urlopen(URL, timeout=60) as source, archive.open("wb") as destination:
            shutil.copyfileobj(source, destination)
        with ZipFile(archive) as contents:
            for item in contents.infolist():
                parts = PurePosixPath(item.filename).parts
                if not parts or parts[0] != NAME or ".." in parts:
                    raise ValueError(f"Unexpected path in STT model archive: {item.filename}")
            contents.extractall(ROOT)
        if not (MODEL / "am" / "final.mdl").is_file():
            raise RuntimeError("STT model archive did not contain am/final.mdl")
    finally:
        archive.unlink(missing_ok=True)
    print(f"Ready: {MODEL}")


if __name__ == "__main__":
    main()
