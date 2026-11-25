"""Evaluate the checkpoint on the validation split: accuracy, per-class P/R/F1, confusion matrix, latency."""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import read_manifest  # noqa: E402
from src.inference import Inspector, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "executorch"])
    a = ap.parse_args()
    cfg = load_config()
    ins = Inspector(cfg, backend=a.backend)
    df = read_manifest(ROOT / cfg["dataset"]["demo_manifest" if a.demo else "manifest"])
    df = df[df.split == "val"]
    C = ins.classes
    cm = np.zeros((len(C), len(C)), int)
    lat = []
    for _, r in df.iterrows():
        t = time.perf_counter()
        res = ins.inspect(r["image"], with_cam=False)
        lat.append((time.perf_counter() - t) * 1000)
        cm[C.index(r["label"]), C.index(res["pred_class"])] += 1
    per = {}
    for i, c in enumerate(C):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        p, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        per[c] = {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(2 * p * rc / max(p + rc, 1e-9), 4), "support": int(cm[i].sum())}
    out = {"dataset": ins.meta.get("trained_on"), "n_val": int(cm.sum()), "accuracy": round(float(np.trace(cm) / cm.sum()), 4),
           "macro_f1": round(float(np.mean([v["f1"] for v in per.values()])), 4), "per_class": per,
           "mean_latency_ms": round(float(np.mean(lat)), 2), "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
           "backend": ins.backend, "confusion_matrix": cm.tolist(), "classes": C}
    tag = ("demo" if a.demo else "carinthia_s") + ("_executorch" if a.backend == "executorch" else "")
    (ROOT / "reports" / f"eval_{tag}.json").write_text(json.dumps(out, indent=2))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(C)), C, rotation=45, ha="right"); ax.set_yticks(range(len(C)), C)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix ({out['dataset']})")
    fig.tight_layout(); fig.savefig(ROOT / "reports" / f"confusion_{tag}.png", dpi=120)
    print(json.dumps({k: v for k, v in out.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
