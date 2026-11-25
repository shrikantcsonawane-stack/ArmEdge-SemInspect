"""Short transfer-learning run for MobileNetV3-Small with a HARD 30-minute budget.

Frozen backbone except the last N feature blocks; balanced per-class subset; early stopping;
saves best checkpoint + models/training_log.json with the ACTUAL measured results.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.classifier import build_model, save_checkpoint  # noqa: E402
from src.dataset import SEMDataset, read_manifest  # noqa: E402
from src.inference import load_config  # noqa: E402

HARD_LIMIT_MIN = 30.0


def evaluate(model, loader):
    model.eval()
    correct = n = 0
    with torch.inference_mode():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return correct / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="train on the SYNTHETIC demo manifest")
    a = ap.parse_args()
    cfg = load_config()
    t, d = cfg["training"], cfg["dataset"]
    budget_s = min(float(t["max_minutes"]), HARD_LIMIT_MIN) * 60
    if t.get("num_threads"):
        torch.set_num_threads(int(t["num_threads"]))
    torch.manual_seed(d["seed"])

    manifest = ROOT / (d["demo_manifest"] if a.demo else d["manifest"])
    df = read_manifest(manifest)
    # balanced subset: cap each class
    parts = []
    for (lab, sp), g in df.groupby(["label", "split"]):
        cap = int(d["max_per_class"] * (1 if sp == "train" else d["val_fraction"]))
        parts.append(g.sample(min(len(g), max(cap, 1)), random_state=d["seed"]))
    df = pd.concat(parts)
    classes = sorted(df["label"].unique())
    tr, va = df[df.split == "train"], df[df.split == "val"]
    print(f"Classes: {classes}\nTrain {len(tr)} / Val {len(va)}  | budget {budget_s/60:.0f} min")

    size = cfg["model"]["image_size"]
    tl = DataLoader(SEMDataset(tr, classes, size, augment=True), batch_size=t["batch_size"], shuffle=True, num_workers=2)
    vl = DataLoader(SEMDataset(va, classes, size), batch_size=t["batch_size"], num_workers=2)
    model = build_model(len(classes), pretrained=True, unfreeze_last_blocks=cfg["model"]["unfreeze_last_blocks"])
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in params):,} / {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(params, lr=t["lr"])
    lossf = nn.CrossEntropyLoss()

    start, best, bad, hist, stop = time.time(), -1.0, 0, [], "max epochs"
    ckpt = ROOT / cfg["model"]["checkpoint"]
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, t["epochs"] + 1):
        model.train()
        tot = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            if time.time() - start > budget_s:
                break
        acc = evaluate(model, vl)
        el = time.time() - start
        hist.append({"epoch": ep, "train_loss": tot / len(tr), "val_acc": acc, "elapsed_s": round(el, 1)})
        print(f"epoch {ep}: loss {tot/len(tr):.4f} val_acc {acc:.4f} ({el:.0f}s)")
        if acc > best:
            best, bad = acc, 0
            save_checkpoint(ckpt, model, classes, {"trained_on": "SYNTHETIC demo data" if a.demo else "Carinthia-S subset",
                                                   "val_acc": acc, "epoch": ep, "image_size": size})
        else:
            bad += 1
        if el > budget_s:
            stop = "time budget reached"
            break
        if bad >= t["patience"]:
            stop = "early stopping"
            break

    log = {"dataset": "SYNTHETIC demo" if a.demo else "Carinthia-S subset", "classes": classes,
           "n_train": len(tr), "n_val": len(va), "best_val_acc": best, "stop_reason": stop,
           "total_minutes": round((time.time() - start) / 60, 2), "device": "cpu", "history": hist}
    (ROOT / "models" / "training_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps({k: v for k, v in log.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
