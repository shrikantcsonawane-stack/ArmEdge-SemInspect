# ArmEdge-SemInspect: Technical Report (Phase 1)
*On-Device AI Semiconductor Wafer Defect Detection and Quality-Control Assistant*
Bharat AI-SoC Challenge 2026–27, Problem Statement 5

**Status legend:** ✅ Implemented & tested · 🟡 Implemented, not yet tested end-to-end · ⏳ Pending

## 1. Abstract
ArmEdge-SemInspect applies PS5 (real-time industrial defect detection and QC assistance) to semiconductor wafer surface inspection on scanning-electron-microscope (SEM) images. The pipeline runs fully on device:
- OpenCV preprocessing.
- A transfer-learned MobileNetV3-Small classifier, run in PyTorch or exported to ExecuTorch.
- Transparent rule-based severity and PASS/REVIEW/FAIL logic.
- An operator advisor built around Llama 3.2 1B Instruct on ExecuTorch.
- SQLite logging, behind a Streamlit dashboard.

In this phase the classifier was trained and evaluated **only on a synthetic SEM-like demo set**, because the Carinthia-S download was blocked (HTTP 403) on our network. No results on real SEM data are claimed.

## 2. Problem statement (PS5)
The system must:
- detect surface defects and anomalies;
- classify defect type and severity;
- give operators actionable recommendations;
- log inspection results;
- work locally on an edge device without cloud dependence.

Mandatory technologies: Llama and ExecuTorch. Target: an Arm-based edge device such as the Raspberry Pi 5.

## 3. Motivation: wafer inspection
SEM review of wafer defects (particles, scratches, pattern bridges, voids, and so on) is a high-volume, expert-limited step in semiconductor manufacturing. A low-cost edge assistant next to the review tool can pre-classify images, prioritise the human review queue, keep a traceable log, and give consistent dispositions. It does this without sending proprietary fab images to the cloud.

## 4. Proposed solution
SEM image → preprocess → CNN class + confidence → localisation view → severity → QC decision → advisor text → log/dashboard.

The vision output is the single source of truth. The language model only verbalises it.

## 5. System architecture
See `docs/architecture.png`. Modules in `src/`:

| Module | Role |
|---|---|
| `preprocessing` | Loading, denoising, CLAHE, normalisation, overlays |
| `classifier` | MobileNetV3-Small |
| `inference` | Pipeline and backends (PyTorch / ExecuTorch), Grad-CAM |
| `severity` | Severity rules |
| `quality_control` | PASS/REVIEW/FAIL decision |
| `advisor` | Advisor orchestration and fallback |
| `executorch_advisor` | Llama adapter |
| `database` | SQLite log |

All thresholds are in `config/config.yaml`.

## 6. Dataset
**Target: Carinthia-S** (Zenodo 10.5281/zenodo.16895427, CC-BY-4.0). The record states "4,591 images, each paired with a segmentation mask, unevenly distributed across six defect classes", with expert-validated binary masks.

⏳ The archive (`data.zip`, 139 MB) could not be downloaded: Zenodo answered HTTP 403 ("unusual traffic") from our network. As a result:
- class names, folder layout and mask format are **unverified**;
- `scripts/prepare_dataset.py` takes labels from the folder names instead of hard-coding them, and prints the inventory for manual checking.

The record mentions no severity labels.

**Used in this phase: synthetic demo set** (`prepare_dataset.py --demo`):
- 5 classes × 80 images, 256×256: `demo_clean`, `demo_particle`, `demo_scratch`, `demo_bridge`, `demo_void`.
- Each image is a periodic line/space pattern with Gaussian noise and one drawn defect, plus an exact mask.
- These are **not real SEM images**. They serve only to exercise the pipeline.

## 7. Model and training approach ✅
- ImageNet-pretrained MobileNetV3-Small (1.52 M parameters). The backbone is frozen except the last 2 feature blocks; a new linear head is added. Trainable: 946,517 parameters.
- Balanced subset (≤150 images per class), 80/20 stratified split, flip/rot90 augmentation.
- AdamW (lr 1e-3), batch size 32, up to 25 epochs, early stopping with patience 5. The best checkpoint is saved.
- **Time budget:** configurable `max_minutes` (20). `train.py` enforces a hard 30-minute cap and checks it every batch.
- No hyperparameter search, no competing models, no segmentation network, and no Llama training.

## 8. Defect classification ✅
The output is the softmax class and its confidence, plus a Grad-CAM saliency map from the last feature block.

The dashboard labels the two localisation views separately:
- **Grad-CAM** is labelled "model attention, NOT a segmentation".
- A **dataset mask**, when one exists, is labelled "GROUND-TRUTH (dataset annotation), NOT a model prediction".

No predicted segmentation mask is produced.

## 9. Severity estimation ✅ (heuristic)
- **When a valid mask exists**, severity comes from the defect area: below `area_medium_pct` (1%) is LOW, below `area_high_pct` (5%) is MEDIUM, otherwise HIGH.
- **Without a mask**, severity is `default_class_severity` (MEDIUM).
- **Class rules:** optional per-class minimum severity (`class_severity`).
- **Low confidence:** below 0.50, severity is never LOW (conservative).

Every decision exposes its reasons in the UI. These rules are engineering defaults. They are **not validated against expert severity labels**, because none are available.

## 10. Quality decision engine ✅
- LOW → PASS, MEDIUM → REVIEW, HIGH → FAIL (configurable).
- Confidence below `min_confidence` (0.60) escalates a decision to at least REVIEW.
- Secondary inspection is required whenever the decision is not PASS.

## 11. Llama and ExecuTorch integration
**ExecuTorch, vision ✅ tested.** `python -m src.inference --export-pte` runs the following, with ExecuTorch 1.6.0:
1. `torch.export`
2. `to_edge_transform_and_lower` with `XnnpackPartitioner`
3. writes `models/classifier.pte`

The app runs the file through `executorch.runtime.Runtime`. Compared with eager PyTorch, the maximum logit difference is 5.0e-6, and accuracy is identical.

**Llama 3.2 1B Instruct 🟡 implemented, ⏳ not executed.**
- `src/executorch_advisor.py` builds a Llama-3 chat prompt containing only the structured vision facts (JSON), runs it through ExecuTorch's `TextLLMRunner` (shipped in the pip wheel) at temperature 0, and returns four labelled lines: explanation, severity rationale, action, and secondary inspection.
- The structured fields are always displayed from the vision output, not from the LLM.
- The Llama `.pte` has not been exported, because the weights are licence-gated and there was little time. Export steps are in `models/README.md`.
- Until that is done, the UI shows "ExecuTorch model not configured", and recommendations come from a deterministic template labelled **"NOT generated by Llama"**.

## 12. Edge deployment
CPU-only. No CUDA and no network are needed at inference time. All dependencies publish aarch64 wheels. Deployment steps are in the README.

⏳ **Not yet run on a Raspberry Pi 5 or any Arm64 device.** All numbers below come from an x86_64 laptop.

## 13. Evaluation metrics
- Accuracy, per-class precision/recall/F1 and macro-F1, with a confusion matrix (`scripts/evaluate.py`).
- Per-image latency, mean and p95, covering preprocessing and classification.
- In the live simulation: compute FPS and end-to-end FPS.

## 14. Actual results (synthetic demo set only)
Hardware: Intel i5-13420H (x86_64), 12 threads, CPU only, torch 2.15.0, ExecuTorch 1.6.0.

| Item | Measured |
|---|---|
| Training time | **0.53 min** (32 s), 15 epochs, early stopping; best at epoch 10 |
| Train / val images | 320 / 80 (synthetic) |
| Validation accuracy | **0.775** |
| Macro-F1 | **0.782** |
| Per-class F1 | particle 0.968 · scratch 0.968 · void 0.968 · clean 0.556 · bridge 0.452 |
| Latency, PyTorch eager | mean 7.07 ms, p95 7.71 ms |
| Latency, ExecuTorch XNNPACK | mean 3.69 ms, p95 5.22 ms |
| Unit tests | 7 / 7 pass |

The bridge/clean confusion comes from the synthetic generator: a bridge drawn at line intensity is hard to see after blur and noise. **None of these numbers say anything about real Carinthia-S performance.** Raw outputs: `models/training_log.json`, `reports/eval_demo*.json`, `reports/confusion_demo*.png`.

## 15. Limitations
- No training or evaluation on real SEM data yet.
- Severity is heuristic and not expert-validated.
- Localisation uses ground-truth masks or Grad-CAM; there is no learned segmentation.
- The Llama advisor has not run.
- No Arm hardware measurements.
- The live mode replays the dataset; it is not a camera or SEM tool interface.

## 16. Future work
1. Download Carinthia-S, verify its structure, train within the 30-minute budget, and report real metrics.
2. Export Llama 3.2 1B (8da4w) and measure tokens/s on a Raspberry Pi 5.
3. Benchmark on the Pi 5 (ExecuTorch XNNPACK, INT8 quantisation of the classifier).
4. A lightweight segmentation head for predicted defect area.
5. Calibrate confidence and severity thresholds with fab engineers.

## 17. References
See `reports/references.md`.
