"""
performance_estimate.py — NannyML CBPE: estimate model performance without ground truth labels
Problem: labels arrive 6 weeks late (insurance, logistics, credit) — how do you monitor NOW?
Answer: CBPE estimates AUC from predicted probabilities alone (no y_true needed in production)
Run:   python monitoring/performance_estimate.py
"""
import json
from pathlib import Path
import pandas as pd
import nannyml as nml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Path("reports").mkdir(exist_ok=True)

ref_df  = pd.read_csv("model/ref_df.csv")
prod_df = pd.read_csv("model/prod_df.csv")
FEATURES = ["age", "limit_bal", "pay_delay", "bill_amt", "pay_amt", "education"]

estimator = nml.CBPE(
    y_pred_proba="prob",
    y_true="default",
    problem_type="classification_binary",
    metrics=["roc_auc"],
    chunk_size=200,
)
estimator.fit(ref_df)

prod_input = prod_df[FEATURES + ["prob"]].copy()
results = estimator.estimate(prod_input)
df_results = results.to_df()

estimated_aucs = df_results[("roc_auc", "value")].tolist()
alerts = df_results[("roc_auc", "alert")].tolist()

fig, ax = plt.subplots(figsize=(10, 5))
chunks = list(range(len(estimated_aucs)))
ax.plot(chunks, estimated_aucs, "o-", color="#0F3460", linewidth=2, label="Estimated ROC-AUC")
lower = df_results[("roc_auc", "lower_threshold")].iloc[0]
ax.axhline(lower, color="#E94560", linestyle="--", label=f"Alert threshold ({lower:.3f})")
ax.fill_between(chunks, lower, estimated_aucs, alpha=0.15, color="#E94560")
ax.set_title("NannyML CBPE — Estimated ROC-AUC (No Labels Needed)", fontsize=13, pad=12)
ax.set_xlabel("Production Chunk"); ax.set_ylabel("Estimated ROC-AUC")
ax.legend(); ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig("reports/nannyml_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: reports/nannyml_performance.png")

summary = {
    "chunks": len(df_results),
    "estimated_auc_mean": round(float(pd.Series(estimated_aucs).mean()), 4),
    "estimated_auc_min":  round(float(pd.Series(estimated_aucs).min()), 4),
    "alert_threshold":    round(float(lower), 4),
    "alert_detected":     any(alerts),
}
with open("reports/nannyml_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Estimated AUC (mean): {summary['estimated_auc_mean']:.4f}")
print(f"Alert detected:       {summary['alert_detected']}")
