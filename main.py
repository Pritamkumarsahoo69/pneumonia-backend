from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastai.learner import load_learner
from PIL import Image
import io
import matplotlib

matplotlib.use("Agg")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

learn = None  # do NOT load at import time


def get_model():
    global learn
    if learn is None:
        print("Loading model...")
        learn = load_learner("export.pkl")
        print("Model loaded.")
    return learn


@app.get("/")
def home():
    return {"message": "PneumoScan AI Backend Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()

    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pred, pred_idx, probs = model.predict(img)

    return {
        "prediction": str(pred),
        "probabilities": {
            model.dls.vocab[i]: float(probs[i])
            for i in range(len(probs))
        }
    }
