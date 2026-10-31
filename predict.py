"""FastAPI app that serves the trained Ames Housing model."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

HERE = Path(__file__).resolve().parent

with open(HERE / "model.bin", "rb") as f:
    model = pickle.load(f)

with open(HERE / "model_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]

app = FastAPI(title="Ames Housing price prediction")


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["model"], "n_features": len(FEATURES)}


@app.post("/predict")
def predict(payload: dict):
    unknown = set(payload) - set(FEATURES)
    if unknown:
        raise HTTPException(422, f"unknown feature(s): {sorted(unknown)}")

    row = {name: payload.get(name, np.nan) for name in FEATURES}
    df = pd.DataFrame([row])

    try:
        log_price = float(model.predict(df)[0])
    except (TypeError, ValueError) as e:
        raise HTTPException(422, str(e))

    return {
        "predicted_price": round(float(np.expm1(log_price)), 2),
        "predicted_log_price": round(log_price, 4),
        "currency": "USD",
    }
