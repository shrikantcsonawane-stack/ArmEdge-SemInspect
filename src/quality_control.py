"""PASS / REVIEW / FAIL decision engine."""
from __future__ import annotations

ORDER = ["PASS", "REVIEW", "FAIL"]


def decide(severity: str, confidence: float, pred_class: str, cfg: dict) -> dict:
    q = cfg["quality"]
    decision = q["severity_to_decision"][severity]
    reasons = [f"severity {severity} -> {decision}"]
    if confidence < q["min_confidence"] and ORDER.index(decision) < ORDER.index("REVIEW"):
        decision = "REVIEW"
        reasons.append(f"confidence {confidence:.2f} < {q['min_confidence']} -> escalated to REVIEW")
    return {"decision": decision, "reasons": reasons,
            "secondary_inspection": decision != "PASS"}
