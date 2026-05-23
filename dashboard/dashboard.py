"""
dashboard.py — Streamlit monitoring dashboard
Shows: drift status, NannyML estimated AUC, feature distributions
Run: streamlit run dashboard/dashboard.py
"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLOps Monitor", page_icon="📊", layout="wide")
st.title("📊 Credit Default Model — Monitoring Dashboard")
st.caption("Evidently AI + NannyML · Real-time drift detection without waiting for labels")

# ── Load reports ──────────────────────────────────────────────────
drift_path  = Path("reports/drift_metrics.json")
nml_path    = Path("reports/nannyml_summary.json")
nml_img     = Path("reports/nannyml_performance.png")

if not drift_path.exists():
    st.warning("Run `python monitoring/drift_check.py` first to generate reports.")
    st.stop()

with open(drift_path) as f:
    drift = json.load(f)
with open(nml_path) as f:
    nml = json.load(f)

# ── Top KPIs ──────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
drift_detected = drift["dataset_drift_detected"]

col1.metric(
    "Dataset Drift",
    "🔴 DETECTED" if drift_detected else "✅ None",
    delta=f"{drift['number_of_drifted_columns']}/{len(drift['drift_by_column'])} features",
    delta_color="inverse",
)
col2.metric("Drifted Features", f"{drift['number_of_drifted_columns']}")
col3.metric("Est. ROC-AUC", f"{nml['estimated_auc_mean']:.4f}")
col4.metric("Alert Triggered", "⚠️ Yes" if nml["alert_detected"] else "✅ No")

st.divider()

# ── Feature drift status ──────────────────────────────────────────
st.subheader("Feature Drift Status (Evidently AI)")
cols = st.columns(3)
for i, (feature, drifted) in enumerate(drift["drift_by_column"].items()):
    cols[i % 3].metric(
        feature,
        "🔴 DRIFTED" if drifted else "✅ Stable",
        delta_color="inverse",
    )

st.divider()

# ── NannyML chart ─────────────────────────────────────────────────
st.subheader("Estimated Performance Over Time (NannyML CBPE)")
st.caption("No ground truth labels required — CBPE estimates AUC from prediction probabilities")
if nml_img.exists():
    st.image(str(nml_img), use_container_width=True)

st.divider()

# ── Feature distributions ─────────────────────────────────────────
st.subheader("Feature Distributions — Reference vs Production")
ref_df  = pd.read_csv("model/ref_df.csv")
prod_df = pd.read_csv("model/prod_df.csv")
FEATURES = ["age", "limit_bal", "pay_delay", "bill_amt", "pay_amt"]

fig, axes = plt.subplots(1, len(FEATURES), figsize=(16, 4))
for ax, feat in zip(axes, FEATURES):
    ax.hist(ref_df[feat], bins=30, alpha=0.6, label="Reference", color="#0F3460")
    ax.hist(prod_df[feat], bins=30, alpha=0.6, label="Production", color="#E94560")
    ax.set_title(feat, fontsize=9)
    ax.legend(fontsize=7)
plt.suptitle("Reference (training) vs Production distributions", fontsize=11)
plt.tight_layout()
st.pyplot(fig)

st.caption("Built with Evidently AI + NannyML · Deployed on GCP Cloud Run")
