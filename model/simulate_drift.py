"""
simulate_drift.py — Generate realistic drifted production data for monitoring demos
Saves: model/prod_df.csv (drifted), model/prod_df_nodrift.csv (clean)
Run:   python model/simulate_drift.py
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

rng = np.random.default_rng(99)
N = 1000

# --- Drifted production data (older customers, higher balances — covariate shift)
age       = rng.integers(35, 75, N)          # drift: older
limit_bal = rng.normal(200000, 90000, N).clip(10000, 800000)  # drift: higher limits
pay_delay = rng.integers(0, 10, N)           # drift: more late payments
bill_amt  = rng.normal(90000, 50000, N).clip(0, 500000)
pay_amt   = rng.normal(4000, 7000, N).clip(0, 200000)
education = rng.choice([1, 2, 3, 4], N, p=[0.25, 0.45, 0.25, 0.05])

prod_drift = pd.DataFrame({
    "age": age, "limit_bal": limit_bal, "pay_delay": pay_delay,
    "bill_amt": bill_amt, "pay_amt": pay_amt, "education": education,
})

# --- Clean production data (same distribution as training)
age2       = rng.integers(21, 75, N)
limit_bal2 = rng.normal(150000, 80000, N).clip(10000, 800000)
pay_delay2 = rng.integers(-2, 8, N)
bill_amt2  = rng.normal(60000, 40000, N).clip(0, 500000)
pay_amt2   = rng.normal(5000, 8000, N).clip(0, 200000)
education2 = rng.choice([1, 2, 3, 4], N, p=[0.35, 0.40, 0.20, 0.05])

prod_clean = pd.DataFrame({
    "age": age2, "limit_bal": limit_bal2, "pay_delay": pay_delay2,
    "bill_amt": bill_amt2, "pay_amt": pay_amt2, "education": education2,
})

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = ["age", "limit_bal", "pay_delay", "bill_amt", "pay_amt", "education"]
prod_drift["prob"] = model.predict_proba(prod_drift[FEATURES])[:, 1]
prod_clean["prob"] = model.predict_proba(prod_clean[FEATURES])[:, 1]

prod_drift.to_csv("model/prod_df.csv", index=False)
prod_clean.to_csv("model/prod_df_nodrift.csv", index=False)
print(f"Drifted prod:  {len(prod_drift)} rows — avg prob {prod_drift.prob.mean():.3f}")
print(f"Clean prod:    {len(prod_clean)} rows — avg prob {prod_clean.prob.mean():.3f}")
