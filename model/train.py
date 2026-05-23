"""
train.py — LogisticRegression on synthetic credit default dataset
Saves: model.pkl, ref_df.csv, metadata.json
Run:   python model/train.py
"""
import pickle, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

rng = np.random.default_rng(42)
N = 5000

age       = rng.integers(21, 75, N)
limit_bal = rng.normal(150000, 80000, N).clip(10000, 800000)
pay_delay = rng.integers(-2, 8, N)
bill_amt  = rng.normal(60000, 40000, N).clip(0, 500000)
pay_amt   = rng.normal(5000, 8000, N).clip(0, 200000)
education = rng.choice([1, 2, 3, 4], N, p=[0.35, 0.40, 0.20, 0.05])

default = (
    (pay_delay > 2).astype(int) * 0.5
    + (bill_amt / (limit_bal + 1) > 0.7).astype(int) * 0.3
    + rng.random(N) * 0.2
) > 0.45

df = pd.DataFrame({
    "age": age, "limit_bal": limit_bal, "pay_delay": pay_delay,
    "bill_amt": bill_amt, "pay_amt": pay_amt, "education": education,
    "default": default.astype(int),
})

FEATURES = ["age", "limit_bal", "pay_delay", "bill_amt", "pay_amt", "education"]
X, y = df[FEATURES], df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")),
])
pipeline.fit(X_train, y_train)

auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
print(f"Test ROC-AUC: {auc:.4f}")

Path("model").mkdir(exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

ref = X_train.copy()
ref["prob"]    = pipeline.predict_proba(X_train)[:, 1]
ref["default"] = y_train.values
ref.to_csv("model/ref_df.csv", index=False)

meta = {
    "features": FEATURES,
    "roc_auc": round(auc, 4),
    "default_rate": round(float(default.mean()), 3),
}
with open("model/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Saved: model/model.pkl  model/ref_df.csv  model/metadata.json")
