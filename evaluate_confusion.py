# evaluate_confusion.py
import os
from pathlib import Path
from typing import List, Tuple
import requests
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load .env
load_dotenv()

# -----------------------------
# CONFIG
# -----------------------------
TEST_DIR = "dataset/test"  # folder structure: dataset/test/Fresh Apple/*.jpg etc.
PREDICT_MODE = "azure"     # "azure" or "fastapi"

# Azure settings
AZURE_PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
AZURE_PROJECT_ID = os.getenv("AZURE_PROJECT_ID")
AZURE_ITERATION_NAME = os.getenv("AZURE_ITERATION_NAME", "Iteration1")
AZURE_PREDICT_URL = os.getenv("AZURE_PREDICTION_URL")

# FastAPI endpoint
FASTAPI_URL = "http://127.0.0.1:8000/predict"

TOPK = 3

# -----------------------------
# Utilities
# -----------------------------
def iter_images(root: str) -> List[Tuple[str, str]]:
    """Yield (image_path, true_label) from class-subfolder structure."""
    rootp = Path(root)
    for cls_dir in sorted(d for d in rootp.iterdir() if d.is_dir()):
        label = cls_dir.name
        for img in cls_dir.rglob("*"):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                yield str(img), label

def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# -----------------------------
# Prediction functions
# -----------------------------
def predict_azure(image_bytes: bytes) -> List[Tuple[str, float]]:
    headers = {
        "Prediction-Key": AZURE_PREDICTION_KEY,
        "Content-Type": "application/octet-stream",
    }
    r = requests.post(AZURE_PREDICT_URL, headers=headers, data=image_bytes, timeout=30)
    r.raise_for_status()
    data = r.json()
    predictions = data.get("predictions", [])
    return [(p["tagName"], float(p["probability"])) for p in predictions]

def predict_fastapi(image_bytes: bytes) -> List[Tuple[str, float]]:
    """Return list of (label, probability) using your /predict API."""
    files = {"file": ("image.jpg", image_bytes, "application/octet-stream")}
    r = requests.post(FASTAPI_URL, files=files, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Use your app.py logic: find Fresh/Rotten top prediction
    predictions = []
    fruit = data.get("food")
    if fruit:
        predictions.append(("Fresh " + fruit, data.get("fresh_percentage", 0)/100))
        predictions.append(("Rotten " + fruit, data.get("rotten_percentage", 0)/100))
    return predictions

def get_predictions(image_bytes: bytes) -> List[Tuple[str, float]]:
    if PREDICT_MODE == "azure":
        return predict_azure(image_bytes)
    else:
        return predict_fastapi(image_bytes)

# -----------------------------
# Evaluation
# -----------------------------
def main():
    items = list(iter_images(TEST_DIR))
    if not items:
        print(f"No images found under: {TEST_DIR}")
        return

    # Classes from folder names
    classes = sorted({label for _, label in items})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_true_idx = []
    y_pred_idx = []
    topk_hits = 0
    top1_hits = 0

    for path, true_label in items:
        try:
            img_bytes = load_image_bytes(path)
            preds = get_predictions(img_bytes)
            preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)

            top1_label = preds_sorted[0][0] if preds_sorted else None
            topk_labels = [l for l, _ in preds_sorted[:TOPK]]

            y_true_idx.append(class_to_idx[true_label])
            if top1_label in class_to_idx:
                y_pred_idx.append(class_to_idx[top1_label])
            else:
                y_pred_idx.append(-1)

            # Metrics
            if top1_label == true_label:
                top1_hits += 1
            if true_label in topk_labels:
                topk_hits += 1
        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")
            y_true_idx.append(class_to_idx[true_label])
            y_pred_idx.append(-1)

    classes_extended = classes + ["Unknown"]
    y_pred_idx = [p if p >= 0 else len(classes) for p in y_pred_idx]

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(len(classes_extended))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    top1_acc = top1_hits / len(items)
    topk_acc = topk_hits / len(items)
    print(f"\nSamples: {len(items)}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-{TOPK} Accuracy: {topk_acc:.4f}")

    print("\nClassification Report (Top-1):")
    valid_labels = list(range(len(classes)))
    try:
        print(classification_report(y_true_idx, [p if p < len(classes) else -1 for p in y_pred_idx],
                                    labels=valid_labels, target_names=classes, zero_division=0))
    except Exception as e:
        print(f"[INFO] Could not compute classification_report cleanly: {e}")

    # Save confusion matrix images
    out_dir = Path("eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    def plot_cm(matrix, title, fname):
        fig = plt.figure(figsize=(max(6, len(classes_extended)*0.5), max(5, len(classes_extended)*0.5)))
        plt.imshow(matrix, interpolation="nearest")
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes_extended))
        plt.xticks(tick_marks, classes_extended, rotation=45, ha="right")
        plt.yticks(tick_marks, classes_extended)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        fig.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    plot_cm(cm, "Confusion Matrix (counts)", "confusion_counts.png")
    plot_cm(cm_norm, "Confusion Matrix (row-normalized)", "confusion_normalized.png")

    # Save CSV
    np.savetxt(out_dir / "confusion_counts.csv", cm, fmt="%d", delimiter=",")
    np.savetxt(out_dir / "confusion_normalized.csv", cm_norm, fmt="%.6f", delimiter=",")

    print(f"\nSaved to: {out_dir.resolve()}")
    print(" - confusion_counts.png")
    print(" - confusion_normalized.png")
    print(" - confusion_counts.csv")
    print(" - confusion_normalized.csv")


if __name__ == "__main__":
    main()