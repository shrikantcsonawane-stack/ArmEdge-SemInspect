"""Rule-based severity estimation (transparent, configurable).

NOTE: Carinthia-S provides class labels and masks, not expert severity labels.
Severity here is an engineering heuristic, not a learned or validated quantity.
"""
from __future__ import annotations

LEVELS = ["LOW", "MEDIUM", "HIGH"]


def estimate_severity(pred_class: str, confidence: float, area_pct: float | None, cfg: dict) -> dict:
    s = cfg["severity"]
    reasons = []
    if pred_class in s.get("non_defect_classes", []):
        return {"severity": "LOW", "reasons": ["predicted class is a non-defect class"]}

    explicit = s.get("class_severity", {}).get(pred_class)
    if area_pct is not None:
        a = 2 if area_pct >= s["area_high_pct"] else 1 if area_pct >= s["area_medium_pct"] else 0
        reasons.append(f"defect area {area_pct:.2f}% -> {LEVELS[a]} "
                       f"(thresholds {s['area_medium_pct']}% / {s['area_high_pct']}%)")
        level = a
    else:
        level = LEVELS.index(s.get("default_class_severity", "MEDIUM"))
        reasons.append(f"no valid mask -> area not used; default = {LEVELS[level]}")
    if explicit:
        level = max(level, LEVELS.index(explicit))
        reasons.append(f"class rule for '{pred_class}' -> at least {explicit}")
    if confidence < s.get("uncertain_confidence", 0.5):
        level = max(level, 1)  # conservative: an uncertain prediction is never rated LOW
        reasons.append(f"confidence {confidence:.2f} < {s.get('uncertain_confidence', 0.5)} -> at least MEDIUM")
    else:
        reasons.append(f"model confidence = {confidence:.2f}")
    return {"severity": LEVELS[level], "reasons": reasons}
