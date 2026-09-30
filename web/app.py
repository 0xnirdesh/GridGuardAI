"""
GridGuardAI — FastAPI Backend

Endpoints:
  GET  /            → Upload page
  POST /predict     → Receive CSV, run XGBoost, return suspicious customers
  GET  /health      → Health check
"""

import io
import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add src/ to path so we can import feature extractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from train_classifier import extract_features

# ─────────────────────────────────────────────
app = FastAPI(title="GridGuardAI", version="1.0")

app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

# ─────────────────────────────────────────────
# Load XGBoost model once at startup
# ─────────────────────────────────────────────
MODEL_PATH = "models/xgboost.pkl"
THRESHOLD = 0.6

print("Loading model...")
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]
print(f"Model loaded. Features: {len(FEATURE_NAMES)}, threshold: {THRESHOLD}")


# ─────────────────────────────────────────────
# Preprocessing (lightweight version of preprocess.py)
# ─────────────────────────────────────────────
def preprocess_uploaded_csv(df):
    """Convert raw SGCC-format DataFrame to (X_norm, X_raw, ids, flags)."""
    id_col = "CONS_NO"
    label_col = "FLAG"

    if id_col not in df.columns:
        raise ValueError(f"CSV must contain '{id_col}' column")

    has_labels = label_col in df.columns
    date_cols = [c for c in df.columns if c not in (id_col, label_col)]

    flags = df[label_col].astype(int).values if has_labels else np.zeros(len(df), dtype=int)
    ids = df[id_col].values

    consumption = df[date_cols].apply(pd.to_numeric, errors="coerce")
    consumption.index = ids
    consumption.columns = pd.to_datetime(date_cols, format="%m/%d/%Y", errors="coerce")

    # Drop fully-empty days
    bad = consumption.columns[consumption.isna().mean() > 0.95]
    if len(bad):
        consumption = consumption.drop(columns=bad)

    # Interpolate
    consumption = consumption.interpolate(axis=1, method="linear", limit_direction="both")
    consumption = consumption.T.fillna(consumption.mean(axis=1)).T.fillna(0.0)

    # Clip outliers
    mean = consumption.mean(axis=1)
    std = consumption.std(axis=1).replace(0, 1)
    consumption = consumption.clip(lower=mean - 3*std, upper=mean + 3*std, axis=0)

    raw = consumption.values.astype(np.float32)
    mn = consumption.min(axis=1)
    mx = consumption.max(axis=1).replace(0, 1)
    norm = consumption.sub(mn, axis=0).div(mx - mn, axis=0).values.astype(np.float32)

    return norm, raw, ids, flags


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model": "xgboost", "threshold": THRESHOLD}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), low_memory=False)

        if len(df) == 0:
            return JSONResponse({"error": "Empty CSV"}, status_code=400)
        if len(df) > 60000:
            return JSONResponse({"error": "Too many customers (max 60,000)"}, status_code=400)

        # Preprocess
        X_norm, X_raw, ids, flags = preprocess_uploaded_csv(df)

        # Extract 100 features (same as training)
        F, _ = extract_features(X_norm, X_raw)

        # Predict
        probs = MODEL.predict_proba(F)[:, 1]
        preds = (probs > THRESHOLD).astype(int)

        # Build suspicious customers list
        suspicious = []
        for i in range(len(ids)):
            if preds[i] == 1:
                suspicious.append({
                    "customer_id": str(ids[i]),
                    "risk_score": float(probs[i]),
                    "actual_label": int(flags[i]) if len(flags) else None,
                })
        suspicious.sort(key=lambda x: x["risk_score"], reverse=True)

        total = len(ids)
        n_flagged = len(suspicious)
        n_theft_actual = int(flags.sum()) if len(flags) else 0

        elapsed = round(time.time() - t0, 2)

        return JSONResponse({
            "success": True,
            "summary": {
                "total_customers": total,
                "flagged": n_flagged,
                "flagged_pct": round(n_flagged / total * 100, 2),
                "actual_theft": n_theft_actual,
                "threshold": THRESHOLD,
                "processing_time_sec": elapsed,
            },
            "suspicious": suspicious[:200],  # cap at 200 for response size
            "suspicious_total": n_flagged,
        })

    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=400)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
