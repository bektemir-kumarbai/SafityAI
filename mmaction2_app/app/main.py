from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from mmaction.apis import init_recognizer, inference_recognizer

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CONFIG_PATH = BASE_DIR / "checkpoints" / "vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400.py"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth"
LABELS_PATH = BASE_DIR / "app" / "labels.txt"


def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


LABELS = load_labels(LABELS_PATH)

app = FastAPI(
    title="Safety Action API",
    version="1.0.0",
    description="Detect only fight, fall, or normal from video"
)

MODEL = init_recognizer(
    str(CONFIG_PATH),
    str(CHECKPOINT_PATH),
    device="cpu"
)


class DetectResponse(BaseModel):
    action_type: str
    confidence: float
    original_label: str


# Классы, которые будем считать дракой
FIGHT_CLASSES = {
    "wrestling",
    "punching person",
    "kicking",
    "pushing",
    "slapping",
    "massaging person's head",  # можно потом убрать, если будет шум
}

# Классы, которые будем считать падением
FALL_CLASSES = {
    "falling off chair",
    "tumbling",
    "drop kicking",   # спорный, можно убрать
}

# Более мягкая проверка по словам
FIGHT_KEYWORDS = ["wrestling", "punch", "kick", "slap", "push", "hit", "fighting"]
FALL_KEYWORDS = ["fall", "slip", "trip", "tumble"]


def map_to_safety_class(label: str) -> str:
    label_lower = label.lower()

    if label_lower in FIGHT_CLASSES:
        return "fight"

    if label_lower in FALL_CLASSES:
        return "fall"

    for word in FIGHT_KEYWORDS:
        if word in label_lower:
            return "fight"

    for word in FALL_KEYWORDS:
        if word in label_lower:
            return "fall"

    return "normal"


@app.get("/")
def root():
    return {
        "message": "Safety Action API is running",
        "docs": "/docs"
    }


@app.post("/detect", response_model=DetectResponse)
async def detect(video: UploadFile = File(...)):
    allowed_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(video.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        result = inference_recognizer(MODEL, str(temp_path))
        pred_score = result.pred_score
        scores = pred_score.tolist() if hasattr(pred_score, "tolist") else list(pred_score)

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        original_label = LABELS[best_idx] if best_idx < len(LABELS) else f"class_{best_idx}"
        confidence = float(scores[best_idx])

        action_type = map_to_safety_class(original_label)

        return DetectResponse(
            action_type=action_type,
            confidence=round(confidence, 6),
            original_label=original_label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path.exists():
            temp_path.unlink()