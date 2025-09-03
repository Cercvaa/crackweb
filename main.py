import io
import os
import base64
import requests
import pathlib
from pathlib import Path

if hasattr(pathlib, "WindowsPath") and hasattr(pathlib, "PosixPath"):
    pathlib.WindowsPath = pathlib.PosixPath

from typing import Tuple

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from fastai.vision.all import load_learner, PILImage
from PIL import Image

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_LOCAL = MODELS_DIR / "crack_classifier.pkl"
MODEL_URL = os.getenv("MODEL_URL")

def ensure_model():
    if MODEL_LOCAL.exists():
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL env var not set")
    print(f"Downloading model from {MODEL_URL} ...")
    r = requests.get(MODEL_URL, timeout=300)
    r.raise_for_status()
    MODEL_LOCAL.write_bytes(r.content)

ensure_model()
learn = load_learner(MODEL_LOCAL)
vocab = list(learn.dls.vocab)
CRACK_IDX = vocab.index('Positive') if 'Positive' in vocab else 1


def predict_pil(pil: Image.Image) -> Tuple[str, float]:
    """Return ('Crack'/'No Crack', probability for Crack class)."""
    pil = pil.convert("RGB")
    pred, idx, probs = learn.predict(PILImage.create(pil))
    p_crack = float(probs[CRACK_IDX])
    label = "Crack" if (learn.dls.vocab[int(idx)] == 'Positive') else "No Crack"
    return label, p_crack

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
):
    # Read uploaded image into memory
    bytes_data = await file.read()
    try:
        pil = Image.open(io.BytesIO(bytes_data))
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": f"Could not read image: {e}",
            },
            status_code=400,
        )

    label, p_crack = predict_pil(pil)

    # Re-encode the uploaded image for inline display
    img_buf = io.BytesIO()
    pil.save(img_buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "filename": file.filename,
            "label": label,
            "p_crack": f"{p_crack:.3f}",
            "img_b64": img_b64,   # <-- pass base64 string, not raw bytes
        },
    )

# Optional: healthcheck
@app.get("/health")
async def health():
    return {"status": "ok"}