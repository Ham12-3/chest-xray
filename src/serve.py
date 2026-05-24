"""FastAPI inference server.

Loads the best.pt checkpoint and serves predictions on uploaded chest X-ray
images. Returns the 14 finding probabilities and a Grad-CAM heatmap for the
top predicted finding.

IMPORTANT: This is a learning and portfolio project. The output of this API
must never be used to diagnose or treat patients. The model is not clinically
validated.

Run:
    pip install fastapi uvicorn python-multipart
    python -m src.serve

Then POST an image to http://localhost:8000/predict
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import LABELS  # noqa: E402
from src.gradcam import GradCAM  # noqa: E402
from src.model import build_model  # noqa: E402
from src.paths import artifacts_dir  # noqa: E402
from src.preprocess import build_transforms  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Chest X-ray Classifier",
    description=(
        "Predicts 14 findings from a chest X-ray image using DenseNet-121. "
        "NOT a medical device. Not clinically validated. "
        "For educational and portfolio use only."
    ),
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

_model = None
_gradcam = None
_transform = None
_device = None


def _load_model():
    global _model, _gradcam, _transform, _device

    ckpt_path = artifacts_dir() / "best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(
            f"Checkpoint not found at {ckpt_path}. "
            "Run training first: python -m src.train"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {"name": "densenet121", "pretrained": False, "num_classes": 14})

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = build_model(model_cfg).to(_device)
    _model.load_state_dict(ckpt["model"])
    _model.eval()

    _gradcam = GradCAM(_model)

    image_cfg = cfg.get("image", {"size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    _transform = build_transforms(image_cfg, train=False)

    print(f"Model loaded from {ckpt_path} on {_device}")
    print(f"Val macro AUC at checkpoint: {ckpt.get('val_macro_auc', 'unknown'):.4f}")


@app.on_event("startup")
def startup():
    _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a chest X-ray image (PNG or JPEG) and get back 14 finding probabilities.

    Also returns a Grad-CAM heatmap highlighting which region drove the top prediction.

    DISCLAIMER: This output is not a medical diagnosis. The model is not
    clinically validated and must not be used for patient care.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and validate the image.
    contents = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Send a PNG or JPEG.")

    # Run inference.
    tensor = _transform(pil_image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
    probs = torch.sigmoid(logits)[0].cpu().numpy()

    predictions = {label: round(float(p), 4) for label, p in zip(LABELS, probs)}

    # Grad-CAM for the top predicted finding.
    top_idx = int(np.argmax(probs))
    top_label = LABELS[top_idx]
    heatmap = _gradcam.generate(tensor, top_idx)

    # Resize original image to 224 for a consistent overlay size.
    display_size = pil_image.resize((224, 224))
    overlay = _gradcam.overlay(display_size, heatmap)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    return JSONResponse({
        "disclaimer": (
            "NOT a medical diagnosis. This model is not clinically validated "
            "and must never be used for patient care."
        ),
        "predictions": predictions,
        "top_finding": {
            "label": top_label,
            "probability": round(float(probs[top_idx]), 4),
        },
        "gradcam_png_base64": heatmap_b64,
    })


if __name__ == "__main__":
    uvicorn.run("src.serve:app", host="0.0.0.0", port=8000, reload=False)
