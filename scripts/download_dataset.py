"""Download Carinthia-S (Zenodo record 16895427, CC-BY-4.0) into data/carinthia_s/.

Files (from the Zenodo record API): data.zip (139,225,942 B, md5 3a41242574635cf1302fab15accb7730)
and carinthia-s_dataset.html (dataset description). The dataset is NOT committed to git.
If Zenodo is unreachable, download data.zip manually in a browser and pass --zip PATH.
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = "https://zenodo.org/records/16895427/files/"
FILES = {"data.zip": "3a41242574635cf1302fab15accb7730", "carinthia-s_dataset.html": "4d9ca3ed51dfc97444939f742496d2d9"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, dst):
    url = REC + name + "?download=1"
    print("Downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "ArmEdge-SemInspect-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="use an already-downloaded data.zip")
    ap.add_argument("--out", default=str(ROOT / "data" / "carinthia_s"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    zp = Path(a.zip) if a.zip else out / "data.zip"
    try:
        if not a.zip:
            for name in FILES:
                if not (out / name).exists():
                    fetch(name, out / name)
    except Exception as e:
        sys.exit(f"Download failed ({e}). Download data.zip manually from "
                 f"https://zenodo.org/records/16895427 and rerun with --zip PATH.")
    if md5(zp) != FILES["data.zip"]:
        print("WARNING: md5 mismatch for", zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(out / "extracted")
    print("Extracted to", out / "extracted", "\nNext: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
