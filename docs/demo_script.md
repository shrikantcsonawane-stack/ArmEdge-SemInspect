# 2-minute demo script (screen recording)

**Before recording:** `bash scripts/run_demo.sh`, open http://localhost:8501, then run `rm logs/inspections.db` for an empty history.

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:15 | Title + architecture.png | "ArmEdge-SemInspect: an offline, on-device SEM wafer defect inspection and QC assistant for PS5. It uses MobileNetV3-Small, ExecuTorch and a Llama advisor." |
| 0:15–0:25 | Demo-mode banner | "This build is trained on a synthetic SEM-like demo set, because the Carinthia-S download was blocked. The banner says so." |
| 0:25–0:40 | Inspection tab: upload an image (e.g. `data/demo/images/demo_void/demo_void_001.png`), click **Run inspection** | "I upload an SEM image and run the inspection." |
| 0:40–0:55 | Metrics row + visualisation | "Predicted class, confidence and latency appear here. On the right is the defect visualisation: the dataset's ground-truth mask, labelled as such, or Grad-CAM attention." |
| 0:55–1:10 | Severity, decision banner, *Why?* expander | "Severity comes from defect area and confidence using configurable rules, which gives PASS, REVIEW or FAIL. Every rule is shown." |
| 1:10–1:25 | Operator recommendation | "This is the operator recommendation. Its source is shown honestly: the deterministic fallback until the Llama ExecuTorch model is exported." |
| 1:25–1:40 | Live demo tab → **Start live inspection** (20 images) | "This is a dataset-driven live simulation, not a camera feed. Predictions stream in one by one with average latency and FPS." |
| 1:40–1:50 | History tab → **Export CSV** | "Every inspection is logged to SQLite and can be exported as CSV." |
| 1:50–2:00 | System status tab | "The status page shows the vision model, the ExecuTorch backend running the classifier, the Llama status (not configured), the CPU, and offline operation." |
