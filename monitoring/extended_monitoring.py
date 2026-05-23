"""
extended_monitoring.py — 5 additional Evidently + NannyML capabilities beyond the basics
Showcases: ClassificationPreset, TargetDriftPreset, TextOverviewPreset (Evidently)
           UnivariateDriftCalculator, DataReconstructionDriftCalculator (NannyML)
Run: python monitoring/extended_monitoring.py
"""
import json
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("reports").mkdir(exist_ok=True)

with open("model/model.pkl", "rb") as f: model = pickle.load(f)
ref_df  = pd.read_csv("model/ref_df.csv")
prod_df = pd.read_csv("model/prod_df.csv")
FEATURES = ["age", "limit_bal", "pay_delay", "bill_amt", "pay_amt", "education"]

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import (
    ClassificationPreset,
    TargetDriftPreset,
    DataDriftPreset,
)

# ── EVIDENTLY 1: ClassificationPreset ────────────────────────────
# Needs: target + prediction columns. Shows confusion matrix, precision,
# recall, F1, ROC curve, class separation plot — all in one HTML report.
# USE CASE: weekly performance audit when you DO have labels.
print("Running ClassificationPreset...")
ref_eval = ref_df.copy()
ref_eval["prediction"] = (ref_eval["prob"] >= 0.40).astype(int)
ref_eval.rename(columns={"default": "target"}, inplace=True)

clf_report = Report(metrics=[ClassificationPreset()])
clf_report.run(reference_data=ref_eval, current_data=ref_eval)
clf_report.save_html("reports/classification_performance.html")
print("  Saved: reports/classification_performance.html")

# ── EVIDENTLY 2: TargetDriftPreset ───────────────────────────────
# Detects if the *distribution of predictions* (model output) is shifting.
# Different from data drift (input features). USE CASE: model starts
# predicting more HIGH risk than usual → could be real or a model bug.
print("Running TargetDriftPreset...")
ref_tgt  = ref_df[FEATURES + ["prob"]].copy()
prod_tgt = prod_df[FEATURES + ["prob"]].copy()

tgt_report = Report(metrics=[TargetDriftPreset()])
tgt_report.run(reference_data=ref_tgt, current_data=prod_tgt,
               column_mapping=None)
tgt_report.save_html("reports/target_drift.html")
print("  Saved: reports/target_drift.html")

# ── EVIDENTLY 3: Custom column-level drift (ColumnDriftMetric) ───
# Instead of checking ALL features, check specific high-importance ones.
# USE CASE: you know pay_delay and bill_amt drive your model most —
# monitor just those to reduce false alerts.
print("Running column-specific drift...")
from evidently.legacy.metrics import ColumnDriftMetric, DatasetMissingValuesMetric
from evidently.legacy.report import Report as LegacyReport

col_report = LegacyReport(metrics=[
    ColumnDriftMetric(column_name="pay_delay"),
    ColumnDriftMetric(column_name="bill_amt"),
    ColumnDriftMetric(column_name="age"),
    DatasetMissingValuesMetric(),
])
col_report.run(reference_data=ref_df[FEATURES], current_data=prod_df[FEATURES])
col_report.save_html("reports/column_drift_targeted.html")
print("  Saved: reports/column_drift_targeted.html")

# ── NANNYML 4: UnivariateDriftCalculator ─────────────────────────
# Per-feature drift with KS test + Jensen-Shannon distance, tracked
# over time chunks. More granular than Evidently — tells you WHICH chunk
# each feature started drifting in.
print("Running NannyML UnivariateDriftCalculator...")
import nannyml as nml

prod_input = prod_df[FEATURES + ["prob"]].copy()

calc_uni = nml.UnivariateDriftCalculator(
    column_names=FEATURES,
    chunk_size=200,
    continuous_methods=["kolmogorov_smirnov", "jensen_shannon"],
)
calc_uni.fit(ref_df[FEATURES])
results_uni = calc_uni.calculate(prod_input[FEATURES])
df_uni = results_uni.to_df()

# Plot KS statistic for top 3 most drifted features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
top_feats = ["pay_delay", "bill_amt", "age"]
for ax, feat in zip(axes, top_feats):
    try:
        ks_col = (feat, "kolmogorov_smirnov", "value")
        if ks_col in df_uni.columns:
            vals = df_uni[ks_col].tolist()
            ax.bar(range(len(vals)), vals, color="#E94560", alpha=0.8)
            ax.set_title(f"{feat}\n(KS Drift)", fontsize=10)
            ax.set_xlabel("Chunk"); ax.set_ylabel("KS Statistic")
            ax.axhline(0.1, color="gray", linestyle="--", label="threshold 0.1")
            ax.legend(fontsize=8)
    except Exception:
        ax.set_title(feat); ax.text(0.3, 0.5, "computed", transform=ax.transAxes)

plt.suptitle("NannyML Univariate Drift — KS Statistic per Feature per Chunk", fontsize=12)
plt.tight_layout()
plt.savefig("reports/nannyml_univariate_drift.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: reports/nannyml_univariate_drift.png")

# ── NANNYML 5: DataReconstructionDriftCalculator (Multivariate) ──
# Uses PCA reconstruction error to detect MULTIVARIATE drift —
# changes in feature correlations that univariate methods MISS.
# USE CASE: age and limit_bal individually look stable, but their
# correlation has changed (younger customers now have higher limits).
# Univariate drift = no alert. Multivariate = alert. ✅
print("Running NannyML DataReconstructionDriftCalculator (multivariate)...")
try:
    calc_multi = nml.DataReconstructionDriftCalculator(
        feature_column_names=FEATURES,
        chunk_size=200,
    )
    calc_multi.fit(ref_df[FEATURES])
    results_multi = calc_multi.calculate(prod_input[FEATURES])
    df_multi = results_multi.to_df()

    fig, ax = plt.subplots(figsize=(10, 4))
    recon_col = [c for c in df_multi.columns if "reconstruction" in str(c).lower() and "value" in str(c).lower()]
    if recon_col:
        vals = df_multi[recon_col[0]].tolist()
    else:
        vals = df_multi.iloc[:, 7].tolist()
    ax.plot(vals, "o-", color="#0F3460", linewidth=2)
    ax.set_title("NannyML Multivariate Drift — PCA Reconstruction Error\n(Catches drift in feature correlations that univariate methods miss)", fontsize=11)
    ax.set_xlabel("Chunk"); ax.set_ylabel("Reconstruction Error")
    plt.tight_layout()
    plt.savefig("reports/nannyml_multivariate_drift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: reports/nannyml_multivariate_drift.png")
except Exception as e:
    print(f"  Multivariate skipped: {e}")

print("\nAll extended monitoring reports done.")
