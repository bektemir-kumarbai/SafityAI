
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import torch
import numpy as np
from mmaction.apis import init_recognizer, inference_recognizer

CHECKPOINT_FILE = './checkpoints/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth'
CONFIG_FILE = './checkpoints/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py'
LABELS_PATH = './checkpoints/label_map_k400.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#Loading Labels
with open(LABELS_PATH, 'r') as f:
    LABELS = [line.strip() for line in f.readlines()]

# to hold our models
ml_models = {}

# Initialize Model
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading TimeSformer model on {DEVICE}...")
    ml_models["recognizer"] = init_recognizer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    yield
    # Clean up resources on shutdown
    ml_models.clear()
    print("Model unloaded.")

app = FastAPI(
    title="SafetyAI Action Recognition API",
    description="API for detecting actions in uploaded video clips using TimeSformer.",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic model to define the exact JSON response structure
class DetectResponse(BaseModel):
    action_type: str # fight, fall, or normal
    confidence: float
    original_label: str

FIGHT_CLASSES = {"wrestling", "punching person", "kicking", "pushing", "slapping", "massaging person's head"}
FALL_CLASSES = {"falling off chair", "tumbling", "drop kicking"}

def map_to_safety_class(label: str) -> str:
    label_lower = label.strip().lower()
    if label_lower in FIGHT_CLASSES:
        return "fight"
    if label_lower in FALL_CLASSES:
        return "fall"
    return "normal"

@app.get("/", summary="Health Check")
def root():
    return {"message": "Safety Action API is running", "docs": "/docs"}

@app.post(
    "/detect",
    response_model=DetectResponse,
    summary="Detect safety-critical actions in a video",
)
async def detect_action(file: UploadFile = File(...)):
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    temp_video_path = f"temp_{file.filename}"

    try:
        #Saving the uploaded file temporarily to disk
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        #Running inference on the saved video
        result = inference_recognizer(ml_models["recognizer"], temp_video_path)
        pred_scores = result.pred_scores.item.cpu().numpy()

        #top predictions
        top_idx = int(np.argmax(pred_scores))
        confidence = float(pred_scores[top_idx])
        original_label = LABELS[top_idx] if top_idx < len(LABELS) else f"ID:{top_idx}"

        action_type = map_to_safety_class(original_label)
        return DetectResponse(action_type=action_type, confidence=round(confidence * 100, 2),original_label=original_label)
    except Exception as e:
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
