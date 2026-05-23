"""
drift_check.py — Evidently AI data drift + quality monitoring
Run:   python monitoring/drift_check.py
Output: reports/drift_report.html, reports/drift_metrics.json
"""
import json
from pathlib import Path
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset

Path("reports").mkdir(exist_ok=True)

FEATURES = ["age", "limit_bal", "pay_delay", "bill_amt", "pay_amt", "education"]

ref_df  = pd.read_csv("model/ref_df.csv")[FEATURES]
prod_df = pd.read_csv("model/prod_df.csv")[FEATURES]

report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])
report.run(reference_data=ref_df, current_data=prod_df)

report.save_html("reports/drift_report.html")

result = report.as_dict()
summary = result["metrics"][0]["result"]  # DataDrift top-level

drift_summary = {
    "dataset_drift_detected": summary.get("dataset_drift", False),
    "number_of_drifted_columns": summary.get("number_of_drifted_columns", 0),
    "share_of_drifted_columns": round(summary.get("share_of_drifted_columns", 0), 3),
    "drift_by_column": {
        col: data.get("drift_detected", False)
        for col, data in summary.get("drift_by_columns", {}).items()
    },
}

with open("reports/drift_metrics.json", "w") as f:
    json.dump(drift_summary, f, indent=2)

print(f"Dataset drift detected: {drift_summary['dataset_drift_detected']}")
print(f"Drifted columns:        {drift_summary['number_of_drifted_columns']}/{len(FEATURES)}")
print(f"Share drifted:          {drift_summary['share_of_drifted_columns']:.0%}")
for col, drifted in drift_summary["drift_by_column"].items():
    print(f"  {col:<15} {'🔴 DRIFT' if drifted else '✅ stable'}")
