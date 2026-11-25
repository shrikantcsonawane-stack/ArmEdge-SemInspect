"""End-to-end inspection: preprocess -> classify -> localise -> severity -> QC decision.

Backends:
  * pytorch   : eager PyTorch on CPU (default)
  * executorch: models/classifier.pte via executorch.runtime (if installed and exported)
Run `python -m src.inference --export-pte` to export the trained classifier to ExecuTorch (XNNPACK).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .classifier import load_checkpoint
from .preprocessing import load_gray, mask_area_pct, to_tensor
from .quality_control import decide
from .severity import estimate_severity

ROOT = Path(__file__).resolve().parents[1]
PTE_PATH = ROOT / "models" / "classifier.pte"


def load_config(path=ROOT / "config" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def executorch_runtime_available() -> bool:
    try:
        from executorch.runtime import Runtime  # noqa: F401
        return True
    except Exception:
        return False


class Inspector:
    def __init__(self, cfg: dict | None = None, backend: str = "auto"):
        self.cfg = cfg or load_config()
        ckpt = ROOT / self.cfg["model"]["checkpoint"]
        if not ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt}. Run scripts/train.py first.")
        self.model, self.classes, self.meta = load_checkpoint(ckpt)
        self.size = self.cfg["model"]["image_size"]
        self.et_method = None
        if backend in ("auto", "executorch") and PTE_PATH.exists() and executorch_runtime_available():
            from executorch.runtime import Runtime
            self.et_method = Runtime.get().load_program(str(PTE_PATH)).load_method("forward")
        if backend == "executorch" and self.et_method is None:
            raise RuntimeError("ExecuTorch backend requested but runtime or classifier.pte unavailable")
        self.backend = "executorch (XNNPACK .pte)" if self.et_method else "pytorch (CPU eager)"

    # --- model ---------------------------------------------------------------
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.et_method is not None:
            return self.et_method.execute([x])[0]
        with torch.inference_mode():
            return self.model(x)

    def gradcam(self, x: torch.Tensor, cls_idx: int) -> np.ndarray:
        """Grad-CAM saliency from the classifier (model attention, NOT a segmentation)."""
        feats = {}
        layer = self.model.features[-1]
        h = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
        x = x.clone().requires_grad_(True)
        out = self.model(x)
        h.remove()
        a = feats["a"]
        g = torch.autograd.grad(out[0, cls_idx], a)[0]
        cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))[0].detach().numpy()
        return cam / (cam.max() + 1e-8)

    # --- pipeline -------------------------------------------------------------
    def inspect(self, image, gt_mask: np.ndarray | None = None, with_cam: bool = True) -> dict:
        gray = image if isinstance(image, np.ndarray) else load_gray(image)
        t0 = time.perf_counter()
        x = to_tensor(gray, self.size)
        probs = torch.softmax(self._logits(x), dim=1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0  # preprocess + classification
        conf, idx = float(probs.max()), int(probs.argmax())
        pred = self.classes[idx]

        area = mask_area_pct(gt_mask)
        sev = estimate_severity(pred, conf, area, self.cfg)
        qc = decide(sev["severity"], conf, pred, self.cfg)
        return {
            "pred_class": pred,
            "confidence": conf,
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
            "area_pct": area,
            "area_source": "ground-truth mask (dataset annotation)" if area is not None else "not available",
            "severity": sev["severity"],
            "severity_reasons": sev["reasons"],
            "decision": qc["decision"],
            "decision_reasons": qc["reasons"],
            "secondary_inspection": qc["secondary_inspection"],
            "latency_ms": latency_ms,
            "backend": self.backend,
            "model_trained_on": self.meta.get("trained_on", "unknown"),
            "cam": self.gradcam(x, idx) if with_cam else None,
        }


def export_pte(cfg: dict) -> Path:
    """Export the trained classifier to an ExecuTorch .pte (XNNPACK delegate)."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import to_edge_transform_and_lower

    model, _, _ = load_checkpoint(ROOT / cfg["model"]["checkpoint"])
    sample = (torch.randn(1, 3, cfg["model"]["image_size"], cfg["model"]["image_size"]),)
    ep = torch.export.export(model, sample)
    prog = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()]).to_executorch()
    PTE_PATH.write_bytes(prog.buffer)
    return PTE_PATH


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-pte", action="store_true")
    ap.add_argument("--image")
    a = ap.parse_args()
    cfg = load_config()
    if a.export_pte:
        print("Exported:", export_pte(cfg))
    if a.image:
        r = Inspector(cfg).inspect(a.image, with_cam=False)
        print({k: v for k, v in r.items() if k != "probabilities"})
