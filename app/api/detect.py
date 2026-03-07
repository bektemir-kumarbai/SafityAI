from fastapi import APIRouter, UploadFile, File
from app.schemas import DetectResponse
from app.services import ActionDetector
import os
import uuid


TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
detector = ActionDetector()

router = APIRouter()
@router.post("/detect", response_model=DetectResponse)
async def detect_actions(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        return {"action_type": "invalid_file"}

    temp_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        action_type = detector.predict(temp_path)
        return {"action_type": action_type}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

