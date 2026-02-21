from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastai.learner import load_learner
from PIL import Image
import io
import uvicorn
import os
import matplotlib

matplotlib.use("Agg")  # Prevent font cache delay

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model at startup...")
learn = load_learner("export.pkl")
print("Model loaded successfully.")

@app.get("/")
def home():
    return {"message": "PneumoScan AI Backend Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pred, pred_idx, probs = learn.predict(img)

    return {
    "prediction": str(pred),
    "probabilities": {
        model.dls.vocab[i]: float(probs[i])
        for i in range(len(probs))
    }
}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)