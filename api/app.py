"""
app.py — FastAPI prediction endpoint with automatic prediction logging
Every prediction is logged to predictions.log (triggers drift monitoring)
Run: uvicorn api.app:app --reload
"""
import csv, json, pickle, time
from datetime import datetime
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

Path("logs").mkdir(exist_ok=True)
LOG_FILE = Path("logs/predictions.csv")

with open("model/model.pkl", "rb") as f:
    MODEL = pickle.load(f)
with open("model/metadata.json") as f:
    META = json.load(f)
FEATURES = META["features"]

app = FastAPI(
    title="Credit Default Monitor API",
    description="Real-time prediction + automatic logging for drift monitoring",
    version="1.0.0",
)


class CustomerRecord(BaseModel):
    age:       int   = Field(..., ge=18, le=100, example=45)
    limit_bal: float = Field(..., gt=0,          example=150000.0)
    pay_delay: int   = Field(..., ge=-2, le=10,  example=3)
    bill_amt:  float = Field(..., ge=0,           example=80000.0)
    pay_amt:   float = Field(..., ge=0,           example=5000.0)
    education: int   = Field(..., ge=1, le=4,    example=2)


class Prediction(BaseModel):
    prob:         float
    risk_level:   str
    model_version: str = "lr-v1"


@app.get("/health")
def health():
    return {"status": "ok", "model": "lr-v1", "roc_auc_train": META["roc_auc"]}


@app.post("/predict", response_model=Prediction)
def predict(record: CustomerRecord):
    try:
        row = pd.DataFrame([record.model_dump()])
        prob = float(MODEL.predict_proba(row[FEATURES])[0, 1])

        risk = (
            "HIGH"   if prob >= 0.70 else
            "MEDIUM" if prob >= 0.40 else
            "LOW"
        )

        # Log for drift monitoring
        log_exists = LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURES + ["prob", "timestamp"])
            if not log_exists:
                writer.writeheader()
            row_log = record.model_dump()
            row_log["prob"] = round(prob, 4)
            row_log["timestamp"] = datetime.utcnow().isoformat()
            writer.writerow(row_log)

        return Prediction(prob=round(prob, 4), risk_level=risk)

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
