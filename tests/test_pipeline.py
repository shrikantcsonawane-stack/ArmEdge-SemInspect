"""Fast unit tests (no dataset or trained checkpoint required)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.advisor import FALLBACK_LABEL, Advisor, facts_from_result  # noqa: E402
from src.classifier import build_model  # noqa: E402
from src.database import InspectionDB  # noqa: E402
from src.inference import load_config  # noqa: E402
from src.preprocessing import mask_area_pct, overlay, to_tensor  # noqa: E402
from src.quality_control import decide  # noqa: E402
from src.severity import estimate_severity  # noqa: E402

CFG = load_config()


def test_preprocess_shape():
    x = to_tensor(np.random.randint(0, 255, (300, 400), np.uint8), 224)
    assert x.shape == (1, 3, 224, 224) and x.dtype == torch.float32


def test_model_forward_and_freeze():
    m = build_model(6, pretrained=False, unfreeze_last_blocks=2).eval()
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 6)
    assert not any(p.requires_grad for p in m.features[0].parameters())
    assert all(p.requires_grad for p in m.classifier.parameters())


def test_severity_area_rules():
    assert estimate_severity("x", 0.9, 0.2, CFG)["severity"] == "LOW"
    assert estimate_severity("x", 0.9, 2.0, CFG)["severity"] == "MEDIUM"
    assert estimate_severity("x", 0.9, 9.0, CFG)["severity"] == "HIGH"
    assert estimate_severity("x", 0.3, 0.2, CFG)["severity"] == "MEDIUM"  # uncertain -> never LOW
    assert estimate_severity("x", 0.9, None, CFG)["severity"] == CFG["severity"]["default_class_severity"]


def test_quality_decision():
    assert decide("LOW", 0.95, "x", CFG)["decision"] == "PASS"
    assert decide("LOW", 0.40, "x", CFG)["decision"] == "REVIEW"
    assert decide("HIGH", 0.99, "x", CFG)["decision"] == "FAIL"


def test_mask_area_and_overlay():
    m = np.zeros((100, 100), np.uint8); m[:10, :10] = 1
    assert abs(mask_area_pct(m) - 1.0) < 1e-6
    assert overlay(np.zeros((100, 100), np.uint8), mask=m).shape == (100, 100, 3)


def test_advisor_fallback_is_labelled():
    r = {"pred_class": "demo_void", "confidence": 0.8, "area_pct": 2.0, "area_source": "gt", "severity": "MEDIUM",
         "severity_reasons": ["a"], "decision": "REVIEW", "decision_reasons": ["b"], "secondary_inspection": True}
    cfg = {**CFG, "advisor": {**CFG["advisor"], "model_pte": "models/__none__.pte"}}
    out = Advisor(cfg).advise(r)
    assert out["source"].startswith(FALLBACK_LABEL) and "demo_void" in out["text"]
    assert facts_from_result(r)["quality_decision"] == "REVIEW"


def test_database_roundtrip(tmp_path):
    db = InspectionDB(str(tmp_path / "t.db"))
    db.log({"image_name": "a.png", "pred_class": "c", "decision": "PASS"})
    h = db.history()
    assert len(h) == 1 and h.iloc[0]["decision"] == "PASS"
