import os
import uuid
import requests
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
from jose import JWTError, jwt
from bson import ObjectId
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
load_dotenv()

AZURE_PREDICTION_KEY  = os.getenv("AZURE_PREDICTION_KEY")
PREDICT_URL           = os.getenv("AZURE_PREDICTION_URL")
MONGO_URI             = os.getenv("MONGO_URI")
SECRET_KEY            = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM             = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24   # 24 hours

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER         = os.getenv("AZURE_STORAGE_CONTAINER", "scan-images")

# Blob service client
blob_service_client = None


def get_blob_service():
    global blob_service_client

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is missing in Azure App Settings")

    if blob_service_client is None:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)

    return blob_service_client

# ─────────────────────────────────────────────
# App & DB
# ─────────────────────────────────────────────
app = FastAPI(title="SpoilSense API", version="3.0")

client = AsyncIOMotorClient(MONGO_URI)
db     = client["spoilsense"]

users_col   = db["users"]
history_col = db["scan_history"]

# ─────────────────────────────────────────────
# Security helpers
# ─────────────────────────────────────────────
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await users_col.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise credentials_exception
    return user


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ScanHistoryItem(BaseModel):
    food: Optional[str]
    fresh_percentage: float
    rotten_percentage: float
    verdict: str
    image_name: str
    scanned_at: datetime


class FeedbackRequest(BaseModel):
    scan_id: str
    correct_label: str   # "Fresh" | "Semi-Fresh" | "Spoiled"
    comment: Optional[str] = None


# ─────────────────────────────────────────────
# Freshness helpers
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
        tag  = p["tagName"].lower()
        prob = p["probability"]
        if "fresh" in tag and prob > max_fresh:
            fruit = tag.replace("fresh", "").strip()
            fresh_prob = prob
            for r in data.get("predictions", []):
                if "rotten" in r["tagName"].lower() and fruit in r["tagName"].lower():
                    rotten_prob = r["probability"]
            max_fresh = prob

    if fruit is None:
        return {"food": None, "fresh_percentage": 0.0, "rotten_percentage": 0.0, "verdict": "Unknown"}

    fresh_pct  = round(fresh_prob  * 100, 2)
    rotten_pct = round(rotten_prob * 100, 2)

    return {
        "food": fruit,
        "fresh_percentage": fresh_pct,
        "rotten_percentage": rotten_pct,
        "verdict": get_verdict(fresh_pct),
    }



# ─────────────────────────────────────────────
# Azure Blob Storage helper
# ─────────────────────────────────────────────
async def upload_image_to_blob(image_bytes: bytes, original_filename: str) -> str:
    """Upload image to Azure Blob Storage and return public URL."""
    ext       = original_filename.rsplit(".", 1)[-1] if "." in original_filename else "jpg"
    blob_name = f"{uuid.uuid4()}.{ext}"   # unique filename e.g. a1b2c3d4.jpg

    service = get_blob_service()
    container_client = service.get_container_client(AZURE_STORAGE_CONTAINER)
    container_client.upload_blob(name=blob_name, data=image_bytes, overwrite=True)

    # Build public URL
    account_name = blob_service_client.account_name
    url = f"https://{account_name}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER}/{blob_name}"
    return url


# ─────────────────────────────────────────────
# ① AUTH — Register
# ─────────────────────────────────────────────
@app.post("/auth/register", status_code=201, summary="Register a new user")
async def register(body: RegisterRequest):
    existing = await users_col.find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    user_doc = {
        "name":            body.name,
        "email":           body.email,
        "hashed_password": hash_password(body.password),
        "created_at":      datetime.utcnow(),
    }
    result = await users_col.insert_one(user_doc)
    return {"message": "Account created successfully.", "user_id": str(result.inserted_id)}


# ─────────────────────────────────────────────
# ② AUTH — Login
# ─────────────────────────────────────────────
@app.post("/auth/login", response_model=TokenResponse, summary="Login and receive JWT")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_col.find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")

    token = create_access_token({"sub": str(user["_id"])})
    return {"access_token": token, "token_type": "bearer"}


# ─────────────────────────────────────────────
# ③ PREDICT — with optional history save
# ─────────────────────────────────────────────
@app.post("/predict", summary="Predict food freshness (no login required)")
async def predict_route(
    file: UploadFile = File(...),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG supported.")

    contents = await file.read()
    result   = predict_freshness(contents)
    return result


# ─────────────────────────────────────────────
# ④ HISTORY — Save a scan
# ─────────────────────────────────────────────
@app.post("/history", status_code=201, summary="Save a scan to history (requires login)")
async def save_scan(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG supported.")

    contents = await file.read()

    # 1. Upload image to Azure Blob Storage → get back a URL
    image_url = await upload_image_to_blob(contents, file.filename)

    # 2. Run AI prediction
    result = predict_freshness(contents)

    # 3. Save prediction + image URL to MongoDB
    scan_doc = {
        "user_id":           str(current_user["_id"]),
        "food":              result["food"],
        "fresh_percentage":  result["fresh_percentage"],
        "rotten_percentage": result["rotten_percentage"],
        "verdict":           result["verdict"],
        "image_url":         image_url,        # ← URL not just name
        "scanned_at":        datetime.utcnow(),
    }
    inserted = await history_col.insert_one(scan_doc)

    return {
        "scan_id":    str(inserted.inserted_id),
        "prediction": result,
        "image_url":  image_url,
        "message":    "Scan saved to history.",
    }


# ─────────────────────────────────────────────
# ⑤ HISTORY — Get user scan history
# ─────────────────────────────────────────────
@app.get("/history", summary="Get current user's scan history")
async def get_history(
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
):
    cursor = history_col.find(
        {"user_id": str(current_user["_id"])},
        {"_id": 1, "food": 1, "fresh_percentage": 1, "rotten_percentage": 1,
         "verdict": 1, "image_name": 1, "scanned_at": 1,"image_url":1}
    ).sort("scanned_at", -1).limit(limit)

    scans = []
    async for doc in cursor:
        doc["scan_id"] = str(doc.pop("_id"))
        scans.append(doc)

    return {"total": len(scans), "scans": scans}


# ─────────────────────────────────────────────
# ⑥ HISTORY — Delete a scan
# ─────────────────────────────────────────────
@app.delete("/history/{scan_id}", summary="Delete a scan from history")
async def delete_scan(
    scan_id: str,
    current_user: dict = Depends(get_current_user),
):
    result = await history_col.delete_one(
        {"_id": ObjectId(scan_id), "user_id": str(current_user["_id"])}
    )
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return {"message": "Scan deleted."}


# ─────────────────────────────────────────────
# ⑦ BONUS — Food Safety Tips  (suggested feature #1)
#    Returns safety advice based on verdict
# ─────────────────────────────────────────────
SAFETY_TIPS = {
    "Fresh": {
        "message": "This item looks fresh and safe to consume.",
        "tips": [
            "Store in a cool, dry place or refrigerate to maintain freshness.",
            "Consume within the recommended timeframe for best quality.",
        ],
        "safe_to_eat": True,
    },
    "Semi-Fresh": {
        "message": "This item is past its peak but may still be usable.",
        "tips": [
            "Use immediately — do not store for later.",
            "Cook thoroughly before consuming.",
            "Check for unusual smell or texture before eating.",
        ],
        "safe_to_eat": True,
    },
    "Spoiled": {
        "message": "⚠️ This item appears spoiled. Do not consume.",
        "tips": [
            "Discard immediately to avoid foodborne illness.",
            "Do not cook or attempt to salvage spoiled food.",
            "Clean the storage area to prevent cross-contamination.",
        ],
        "safe_to_eat": False,
    },
    "Unknown": {
        "message": "Could not determine freshness. Proceed with caution.",
        "tips": ["Inspect manually before consuming."],
        "safe_to_eat": None,
    },
}


@app.get("/tips/{verdict}", summary="Get food safety tips based on freshness verdict")
async def get_safety_tips(verdict: str):
    # Case-insensitive match against known keys
    key_map = {k.lower(): k for k in SAFETY_TIPS}
    key = key_map.get(verdict.strip().lower())
    if not key:
        raise HTTPException(status_code=400, detail="Invalid verdict. Use Fresh, Semi-Fresh, or Spoiled.")
    return SAFETY_TIPS[key]


# ─────────────────────────────────────────────
# ⑧ BONUS — User Stats  (suggested feature #2)
#    Shows scan breakdown for the current user
# ─────────────────────────────────────────────
@app.get("/stats", summary="Get personalised scan statistics for current user")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    pipeline = [
        {"$match": {"user_id": str(current_user["_id"])}},
        {"$group": {
            "_id":        "$verdict",
            "count":      {"$sum": 1},
            "avg_fresh":  {"$avg": "$fresh_percentage"},
        }},
    ]
    cursor = history_col.aggregate(pipeline)
    breakdown = {}
    total = 0
    async for doc in cursor:
        breakdown[doc["_id"]] = {
            "count":     doc["count"],
            "avg_fresh": round(doc["avg_fresh"], 1),
        }
        total += doc["count"]

    # Most scanned food
    food_pipeline = [
        {"$match": {"user_id": str(current_user["_id"]), "food": {"$ne": None}}},
        {"$group": {"_id": "$food", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1},
    ]
    top_food_cursor = history_col.aggregate(food_pipeline)
    top_food = None
    async for doc in top_food_cursor:
        top_food = doc["_id"]

    return {
        "user":         current_user["name"],
        "total_scans":  total,
        "breakdown":    breakdown,
        "most_scanned": top_food,
    }


# ─────────────────────────────────────────────
# ⑨ FEEDBACK — Submit correction on a scan
# ─────────────────────────────────────────────
@app.post("/feedback", status_code=201, summary="Submit feedback to correct an AI prediction")
async def submit_feedback(
    body: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
):
    scan = await history_col.find_one(
        {"_id": ObjectId(body.scan_id), "user_id": str(current_user["_id"])}
    )
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")

    await history_col.update_one(
        {"_id": ObjectId(body.scan_id)},
        {"$set": {
            "user_feedback": {
                "correct_label": body.correct_label,
                "comment":       body.comment,
                "submitted_at":  datetime.utcnow(),
            }
        }},
    )
    return {"message": "Thank you! Your feedback helps improve SpoilSense."}


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0"}
