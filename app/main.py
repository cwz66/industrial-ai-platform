import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from functools import lru_cache
from app.model import ImageClassifier

app = FastAPI(title="Industrial AI Platform")

@app.get("/")
def home():
    return {"message": "Industrial AI Platform is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "0.1.0"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")

    max_size = 5 * 1024 * 1024
    content = await file.read(max_size + 1)

    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File size exceeds the maximum limit of 5MB.")

    array = np.frombuffer(content, np.uint8)

    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))

    return {
            "filename": file.filename,
            "width": width,
            "height": height,
            "mean_brightness": round(brightness, 2)
    }

@lru_cache(maxsize=1)
def get_classifier():
    return ImageClassifier()

@app.post("/predict")
def predict_image(file: UploadFile = File(...), top_k: int = Query(default=5, ge=1, le=10)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG are allowed.")

    max_size = 5 * 1024 * 1024
    content = file.file.read(max_size + 1)

    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File size exceeds the maximum limit of 5MB.")

    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    classifier = get_classifier()
    predictions = classifier.predict(image, top_k=top_k)

    return {"filename": file.filename,
            "predictions": predictions}
