import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from dotenv import load_dotenv

load_dotenv()

AZURE_PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
PREDICT_URL = os.getenv("AZURE_PREDICTION_URL")

app = FastAPI(title="SpoilSense API (Predict Only)", version="1.0")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def get_verdict(fresh_pct: float) -> str:
    if fresh_pct >= 70:
        return "Fresh"
    elif fresh_pct >= 40:
        return "Semi-Fresh"
    else:
        return "Spoiled"


def predict_freshness(image_bytes: bytes) -> dict:
    headers = {
        "Prediction-Key": AZURE_PREDICTION_KEY,
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(PREDICT_URL, headers=headers, data=image_bytes, timeout=20)
    response.raise_for_status()
    data = response.json()

    fruit = None
    fresh_prob = 0.0
    rotten_prob = 0.0
    max_fresh = 0.0

    for p in data.get("predictions", []):
        tag = p["tagName"].lower()
        prob = p["probability"]

        if "fresh" in tag and prob > max_fresh:
            fruit = tag.replace("fresh", "").strip()
            fresh_prob = prob

            for r in data.get("predictions", []):
                if "rotten" in r["tagName"].lower() and fruit in r["tagName"].lower():
                    rotten_prob = r["probability"]

            max_fresh = prob

    if fruit is None:
        return {
            "food": None,
            "fresh_percentage": 0.0,
            "rotten_percentage": 0.0,
            "verdict": "Unknown"
        }

    fresh_pct = round(fresh_prob * 100, 2)
    rotten_pct = round(rotten_prob * 100, 2)

    return {
        "food": fruit,
        "fresh_percentage": fresh_pct,
        "rotten_percentage": rotten_pct,
        "verdict": get_verdict(fresh_pct),
    }


# ─────────────────────────────────────────────
# MAIN PREDICT ENDPOINT
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG supported.")

    image_bytes = await file.read()

    result = predict_freshness(image_bytes)

    return {
        "success": True,
        "prediction": result
    }


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}
