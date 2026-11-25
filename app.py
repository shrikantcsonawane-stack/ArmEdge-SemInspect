"""ArmEdge-SemInspect — Streamlit dashboard (runs fully offline on CPU)."""
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import streamlit as st
import torch

from src.advisor import Advisor
from src.database import InspectionDB
from src.dataset import read_manifest
from src.inference import PTE_PATH, ROOT, Inspector, executorch_runtime_available, load_config
from src.preprocessing import load_gray, load_mask, overlay

st.set_page_config(page_title="ArmEdge-SemInspect", page_icon="🔬", layout="wide")
DECISION_COLOR = {"PASS": "#1b8a3a", "REVIEW": "#c98a00", "FAIL": "#c62828"}


@st.cache_resource
def resources():
    cfg = load_config()
    try:
        ins, err = Inspector(cfg), None
    except Exception as e:
        ins, err = None, str(e)
    return cfg, ins, err, Advisor(cfg), InspectionDB(str(ROOT / cfg["database"]["path"]))


@st.cache_data
def manifest():
    cfg = load_config()
    for key in ("manifest", "demo_manifest"):
        p = ROOT / cfg["dataset"][key]
        if p.exists():
            return read_manifest(p), key
    return None, None


cfg, inspector, load_err, advisor, db = resources()
mdf, mkey = manifest()

st.title("🔬 ArmEdge-SemInspect")
st.caption("On-device AI semiconductor wafer (SEM) defect detection & quality-control assistant · "
           "Bharat AI-SoC Challenge 2026–27 · PS5")
if inspector is None:
    st.error(f"Vision model not loaded: {load_err}")
    st.stop()
if "SYNTHETIC" in str(inspector.meta.get("trained_on", "")).upper():
    st.warning("**Demo mode:** the loaded classifier was trained on a SYNTHETIC SEM-like demo set "
               "(classes `demo_*`), not on Carinthia-S. Predictions on real SEM images are not meaningful.")


def gt_mask_for(name: str):
    """Ground-truth mask from the dataset manifest, matched by file name (if any)."""
    if mdf is None:
        return None
    hit = mdf[mdf["image"].map(lambda p: Path(p).name) == name]
    return load_mask(hit.iloc[0]["mask"]) if len(hit) and hit.iloc[0]["mask"] else None


def show_result(gray, res, mask, adv, key=""):
    c1, c2 = st.columns(2)
    c1.image(gray, caption="Original SEM image", width="stretch", clamp=True)
    if mask is not None:
        c2.image(overlay(gray, mask=mask), width="stretch",
                 caption="GROUND-TRUTH defect mask (dataset annotation) — NOT a model prediction")
    else:
        c2.image(overlay(gray, heat=res["cam"]), width="stretch",
                 caption="Model attention (Grad-CAM from classifier) — saliency, NOT a segmentation mask")
    m = st.columns(5)
    m[0].metric("Defect class", res["pred_class"])
    m[1].metric("Confidence", f"{res['confidence']*100:.1f}%")
    m[2].metric("Severity", res["severity"])
    m[3].metric("Defect area", "n/a" if res["area_pct"] is None else f"{res['area_pct']:.2f}%")
    m[4].metric("Inference latency", f"{res['latency_ms']:.1f} ms")
    col = DECISION_COLOR[res["decision"]]
    st.markdown(f"<div style='padding:10px;border-radius:8px;background:{col};color:white;font-size:1.4em;"
                f"text-align:center'><b>QC DECISION: {res['decision']}</b></div>", unsafe_allow_html=True)
    with st.expander("Why? (transparent rules)"):
        st.write("**Severity:**", "; ".join(res["severity_reasons"]))
        st.write("**Decision:**", "; ".join(res["decision_reasons"]))
        st.write("**Area source:**", res["area_source"])
        st.bar_chart(pd.Series(res["probabilities"], name="probability"))
    st.subheader("Operator recommendation")
    st.caption(f"Source: {adv['source']}")
    st.code(adv["text"], language=None)


def record(name, res, adv, mode):
    db.log({"image_name": name, "mode": mode, "pred_class": res["pred_class"],
            "confidence": f"{res['confidence']:.4f}", "severity": res["severity"], "decision": res["decision"],
            "area_pct": None if res["area_pct"] is None else f"{res['area_pct']:.3f}",
            "area_source": res["area_source"], "latency_ms": f"{res['latency_ms']:.2f}",
            "advisor_source": adv["source"], "recommendation": adv["text"].replace("\n", " | "),
            "model_trained_on": res["model_trained_on"]})


tab_i, tab_l, tab_h, tab_s = st.tabs(["Inspection", "Live demo", "History", "System status"])

# ---------------- A. Inspection ----------------
with tab_i:
    up = st.file_uploader("Upload SEM image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    up_mask = st.file_uploader("Optional: ground-truth mask for this image", type=["png", "jpg", "bmp", "tif"])
    if mdf is not None:
        sample = st.selectbox("…or pick a dataset sample", ["—"] + [Path(p).name for p in mdf[mdf.split == "val"]["image"]])
    else:
        sample = "—"
    if st.button("Run inspection", type="primary"):
        if up is not None:
            name, gray = up.name, load_gray(up.getvalue())
            mask = (load_gray(up_mask.getvalue()) > 0).astype(np.uint8) if up_mask else gt_mask_for(up.name)
        elif sample != "—":
            path = mdf[mdf["image"].map(lambda p: Path(p).name) == sample].iloc[0]["image"]
            name, gray, mask = sample, load_gray(path), gt_mask_for(sample)
        else:
            st.info("Upload an image or choose a sample first.")
            st.stop()
        if mask is not None and mask.shape != gray.shape:
            mask = None
        res = inspector.inspect(gray, gt_mask=mask)
        adv = advisor.advise(res)
        record(name, res, adv, "single")
        show_result(gray, res, mask, adv)

# ---------------- B. Live demo ----------------
with tab_l:
    st.info("**Dataset-driven live inspection simulation** — images are replayed from the local dataset "
            "manifest. This is NOT a real SEM camera feed.")
    if mdf is None:
        st.warning("No dataset manifest found. Run scripts/prepare_dataset.py (--demo for synthetic data).")
    else:
        n = st.slider("Number of images", 5, 100, 20)
        delay = st.slider("Display delay between frames (s)", 0.0, 1.0, 0.2)
        log_live = st.checkbox("Log results to history", value=True)
        if st.button("Start live inspection", type="primary"):
            rows = mdf[mdf.split == "val"].sample(n=min(n, (mdf.split == "val").sum()), random_state=int(time.time()))
            stats, frame, table = st.empty(), st.empty(), st.empty()
            lats, out, t_start = [], [], time.perf_counter()
            for i, (_, r) in enumerate(rows.iterrows(), 1):
                gray, mask = load_gray(r["image"]), load_mask(r["mask"])
                res = inspector.inspect(gray, gt_mask=mask, with_cam=False)
                adv = advisor.advise(res)
                lats.append(res["latency_ms"])
                if log_live:
                    record(Path(r["image"]).name, res, adv, "live-sim")
                proc_fps = 1000.0 / np.mean(lats)
                with frame.container():
                    c1, c2 = st.columns([1, 1])
                    vis = overlay(gray, mask=mask) if mask is not None else gray
                    c1.image(vis, clamp=True, width=320, caption=f"Frame {i}/{len(rows)} · {Path(r['image']).name}"
                             + (" · green = GROUND-TRUTH mask" if mask is not None else ""))
                    c2.markdown(f"### {res['pred_class']}  \nConfidence **{res['confidence']*100:.1f}%** · "
                                f"Severity **{res['severity']}**  \n"
                                f"<span style='color:{DECISION_COLOR[res['decision']]};font-size:1.6em'>"
                                f"<b>{res['decision']}</b></span>", unsafe_allow_html=True)
                    c2.caption(f"Ground-truth label: {r['label']}")
                stats.markdown(f"**Avg inference:** {np.mean(lats):.1f} ms · **Model FPS (compute only):** "
                               f"{proc_fps:.1f} · **End-to-end FPS (incl. display/delay):** "
                               f"{i / (time.perf_counter() - t_start):.2f}")
                out.append({"image": Path(r["image"]).name, "true": r["label"], "pred": res["pred_class"],
                            "conf": round(res["confidence"], 3), "severity": res["severity"], "decision": res["decision"]})
                table.dataframe(pd.DataFrame(out[::-1]), width="stretch", height=250)
                time.sleep(delay)
            acc = np.mean([o["true"] == o["pred"] for o in out])
            st.success(f"Processed {len(out)} images · avg {np.mean(lats):.1f} ms · accuracy on this run {acc*100:.1f}%")

# ---------------- C. History ----------------
with tab_h:
    hist = db.history()
    st.write(f"{len(hist)} inspection records (SQLite: `{cfg['database']['path']}`)")
    if len(hist):
        st.dataframe(hist, width="stretch", height=400)
        c = st.columns(3)
        for i, d in enumerate(["PASS", "REVIEW", "FAIL"]):
            c[i].metric(d, int((hist["decision"] == d).sum()))
    st.download_button("Export CSV", hist.to_csv(index=False).encode(), "inspection_history.csv", "text/csv")

# ---------------- D. System status ----------------
with tab_s:
    a_status = advisor.status()
    et_ok = executorch_runtime_available()
    vm = psutil.virtual_memory()
    info = {
        "Vision model": f"MobileNetV3-Small (transfer learning) · classes: {', '.join(inspector.classes)}",
        "Model trained on": inspector.meta.get("trained_on", "unknown"),
        "Checkpoint val. accuracy": f"{inspector.meta.get('val_acc', float('nan')):.3f}",
        "Vision inference backend": inspector.backend,
        "ExecuTorch Python runtime": "installed" if et_ok else "not installed",
        "Classifier .pte (ExecuTorch)": "present" if PTE_PATH.exists() else "not exported",
        "Llama status": a_status["message"] + ("" if a_status["configured"] else f" (missing: {', '.join(a_status['missing'])})"),
        "ExecuTorch LLM runner": a_status.get("runner") or "not available",
        "Advisor backend in use": a_status["backend"],
        "Device": f"{platform.machine()} · {platform.system()} {platform.release()}",
        "CPU": f"{platform.processor() or platform.machine()} · {os.cpu_count()} cores · torch threads {torch.get_num_threads()}",
        "Memory": f"{vm.total/1e9:.1f} GB total · {vm.percent}% used",
        "PyTorch": torch.__version__,
        "CUDA required": "No (CPU inference)",
        "Offline operation": "Yes — no network/cloud calls in the inspection or advisor path",
    }
    st.table(pd.DataFrame(info.items(), columns=["Component", "Status"]).set_index("Component"))
    if not a_status["configured"]:
        st.warning("ExecuTorch model not configured — operator recommendations come from the deterministic "
                   "rule-based fallback and are NOT generated by Llama. See models/README.md to enable it.")
