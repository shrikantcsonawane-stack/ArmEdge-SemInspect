"""Build a manifest CSV (image, mask, label, split).

Real data: scans data/carinthia_s/extracted, PRINTS the discovered structure, and infers
labels from the image's parent folder name. The Carinthia-S archive layout could not be
verified during development (Zenodo was unreachable), so check the printed inventory and
adjust --mask-token / folder logic if needed. Class names are never hard-coded.

--demo: generates a small SYNTHETIC SEM-like dataset (clearly labelled demo_* classes with
exact masks) so the full pipeline can run without the real dataset.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def split(df, val_fraction, seed):
    rng = random.Random(seed)
    df["split"] = "train"
    for _, g in df.groupby("label"):
        idx = list(g.index)
        rng.shuffle(idx)
        df.loc[idx[: max(1, int(len(idx) * val_fraction))], "split"] = "val"
    return df


def real_manifest(raw: Path, mask_token: str) -> pd.DataFrame:
    files = [p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"Found {len(files)} image files under {raw}")
    print("Top-level folders:", Counter(p.relative_to(raw).parts[0] for p in files).most_common(20))
    is_mask = lambda p: mask_token in p.name.lower() or any(mask_token in s.lower() for s in p.parent.parts[-2:])
    masks = {p.stem.lower().replace(mask_token, "").strip("_-"): p for p in files if is_mask(p)}
    rows = []
    for p in files:
        if is_mask(p):
            continue
        m = masks.get(p.stem.lower())
        rows.append({"image": str(p), "mask": str(m) if m else "", "label": p.parent.name})
    df = pd.DataFrame(rows)
    print(f"Images: {len(df)}, with matched masks: {(df['mask'] != '').sum()}")
    print("Labels (from parent folder names):\n", df["label"].value_counts())
    return df


def make_demo(out: Path, n_per_class: int, seed: int) -> pd.DataFrame:
    """Synthetic SEM-like 256x256 images: periodic line/space pattern + noise + one defect type."""
    rng = np.random.default_rng(seed)
    classes = ["demo_clean", "demo_particle", "demo_scratch", "demo_bridge", "demo_void"]
    rows = []
    for c in classes:
        (out / "images" / c).mkdir(parents=True, exist_ok=True)
        (out / "masks" / c).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            s = 256
            pitch = int(rng.integers(14, 24))
            img = np.full((s, s), 70, np.uint8)
            for x in range(0, s, pitch):
                cv2.rectangle(img, (x, 0), (x + pitch // 2, s), 150, -1)
            if rng.random() < 0.5:
                img = img.T.copy()
            mask = np.zeros((s, s), np.uint8)
            cx, cy = (int(v) for v in rng.integers(40, s - 40, 2))
            r = int(rng.integers(5, 22))
            if c == "demo_particle":
                cv2.circle(mask, (cx, cy), r, 1, -1); img[mask > 0] = 235
            elif c == "demo_void":
                cv2.ellipse(mask, (cx, cy), (r, int(r * 0.7)), int(rng.integers(0, 180)), 0, 360, 1, -1); img[mask > 0] = 15
            elif c == "demo_scratch":
                ang = rng.uniform(0, np.pi)
                L = rng.integers(60, 200)
                p1 = (int(cx - L * np.cos(ang) / 2), int(cy - L * np.sin(ang) / 2))
                p2 = (int(cx + L * np.cos(ang) / 2), int(cy + L * np.sin(ang) / 2))
                cv2.line(mask, p1, p2, 1, int(rng.integers(2, 5))); img[mask > 0] = 25
            elif c == "demo_bridge":
                cv2.rectangle(mask, (cx - r, cy - 3), (cx + r, cy + 3), 1, -1)
                mask = mask if img[0, :].std() > 0 else mask.T.copy()  # bridge across lines
                img[mask > 0] = 150
            img = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
            img += rng.normal(0, 18, img.shape)  # SEM shot noise
            img = np.clip(img, 0, 255).astype(np.uint8)
            ip, mp = out / "images" / c / f"{c}_{i:03d}.png", out / "masks" / c / f"{c}_{i:03d}_mask.png"
            cv2.imwrite(str(ip), img)
            cv2.imwrite(str(mp), mask * 255)
            rows.append({"image": str(ip.relative_to(ROOT)), "mask": "" if c == "demo_clean" else str(mp.relative_to(ROOT)), "label": c})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="generate SYNTHETIC demo data instead")
    ap.add_argument("--n-per-class", type=int, default=80)
    ap.add_argument("--mask-token", default="mask")
    ap.add_argument("--raw", default=str(ROOT / "data" / "carinthia_s" / "extracted"))
    a = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config" / "config.yaml"))["dataset"]
    if a.demo:
        df, out = make_demo(ROOT / "data" / "demo", a.n_per_class, cfg["seed"]), ROOT / cfg["demo_manifest"]
    else:
        df, out = real_manifest(Path(a.raw), a.mask_token), ROOT / cfg["manifest"]
    df = split(df, cfg["val_fraction"], cfg["seed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
