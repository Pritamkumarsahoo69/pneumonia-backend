from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastai.learner import load_learner
from PIL import Image
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

learn = load_learner("export.pkl")

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
        "confidence": float(probs[pred_idx])
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)