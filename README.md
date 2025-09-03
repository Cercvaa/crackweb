# Concrete Crack Classifier (FastAPI + fastai)

This is a tiny FastAPI web app that loads a fastai image classifier and predicts **Crack / No Crack** on uploaded or captured photos.

## Quickstart

```bash
# 1) Create & activate a venv (optional but recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Place your exported fastai model here:
#    models/crack_classifier.pkl

# 4) Run the server
uvicorn main:app --reload
# Open http://127.0.0.1:8000
```

## Notes
- The app expects the fastai learner (`crack_classifier.pkl`) to have vocab like `['Negative', 'Positive']`
  where `'Positive'` means **crack**. If your class names differ, adjust the code accordingly.
- Try the **/camera** page for live capture on mobile/desktop (HTTPS or localhost).

## Health check
Visit `/health` to confirm the app is up (also reports if the model file is present).
