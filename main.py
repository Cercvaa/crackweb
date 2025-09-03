# ---- PATCH: allow Windows-exported pickles to load on Linux ----
import pathlib
if hasattr(pathlib, "WindowsPath") and hasattr(pathlib, "PosixPath"):
    pathlib.WindowsPath = pathlib.PosixPath
# ----------------------------------------------------------------

import io, os, base64, requests, pickle, types
from pathlib import Path, PosixPath
from typing import Tuple

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from PIL import Image
from fastai.vision.all import load_learner, PILImage

# ---- FastAPI app + templates/static ----
APP_DIR = Path(__file__).parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="Concrete Crack Classifier")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---- Model paths & lazy download ----
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_LOCAL = MODELS_DIR / "crack_classifier.pkl"
MODEL_URL = os.getenv("MODEL_URL")  # e.g. HF link

def ensure_model():
    if MODEL_LOCAL.exists(): 
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL env var not set")
    print(f"Downloading model from {MODEL_URL} ...")
    r = requests.get(MODEL_URL, timeout=300)
    r.raise_for_status()
    MODEL_LOCAL.write_bytes(r.content)

# ---- Custom unpickler to map WindowsPath -> PosixPath (extra safety) ----
class _PathFixUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name in ("WindowsPath", "_WindowsPath"):
            return PosixPath
        return super().find_class(module, name)

def _pathfix_load(file_obj):
    return _PathFixUnpickler(file_obj).load()

_fix_pickle = types.SimpleNamespace(load=_pathfix_load, Unpickler=_PathFixUnpickler)

# ---- Load model once ----
ensure_model()
learn = load_learner(MODEL_LOCAL, pickle_module=_fix_pickle)
vocab = list(learn.dls.vocab)
CRACK_IDX = vocab.index('Positive') if 'Positive' in vocab else 1

def predict_pil(pil: Image.Image) -> Tuple[str, float]:
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
async def predict(request: Request, file: UploadFile = File(...)):
    bytes_data = await file.read()
    try:
        pil = Image.open(io.BytesIO(bytes_data))
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Could not read image: {e}"},
            status_code=400,
        )

    label, p_crack = predict_pil(pil)

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
            "img_b64": img_b64,
        },
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_present": MODEL_LOCAL.exists()}
