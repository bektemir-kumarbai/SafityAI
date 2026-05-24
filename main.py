import cv2
import torch
import numpy as np
import os
import threading
import asyncio
from collections import deque
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from mmaction.apis import init_recognizer, inference_recognizer
import time
from telegram_service import telegram_service

CHECKPOINT_FILE = './checkpoints/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth'
CONFIG_FILE = './checkpoints/timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py'
LABELS_PATH = './checkpoints/label_map_k400.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Labels
try:
    with open(LABELS_PATH, 'r') as f:
        LABELS = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"Warning: {LABELS_PATH} not found.")
    LABELS = []

FIGHT_CLASSES = {"wrestling", "punching person", "kicking", "pushing", "slapping", "massaging person's head"}
FALL_CLASSES = {"falling off chair", "tumbling", "drop kicking", "fall"}

app = FastAPI(title="SafetyAI Monitoring API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class SystemState:
    def __init__(self):
        self.frame = None
        self.status = "Normal" # or "Alarm"
        self.latest_texts = []
        self.is_inferring = False
        self.buffer_size = 16
        self.frame_buffer = deque(maxlen=self.buffer_size)

state = SystemState()
model = None
loop = None

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def run_inference(frames):
    global state
    try:
        temp_video_path = 'temp_clip.avi'
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(temp_video_path, fourcc, 8, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
            result = inference_recognizer(model, temp_video_path)
            if hasattr(result, 'pred_score'):
                pred_scores = result.pred_score.cpu().numpy()
            elif hasattr(result, 'pred_scores'):
                # Handle mmaction version differences
                pred_scores = result.pred_scores.item.cpu().numpy()
            else:
                pred_scores = result.pred_score.cpu().numpy() # fallback
            
            top_idx = int(np.argmax(pred_scores))
            confidence = float(pred_scores[top_idx])
            action_name = LABELS[top_idx] if top_idx < len(LABELS) else f"ID:{top_idx}"
            
            label_lower = action_name.strip().lower()
            action_type = "normal"
            if label_lower in FIGHT_CLASSES:
                action_type = "fight"
            elif any(fall_kw in label_lower for fall_kw in FALL_CLASSES):
                action_type = "fall"
                
            text_line = f"{action_name}: {confidence * 100:.2f}%"
            state.latest_texts = [text_line]
            
            # Check for alarm
            if action_type == "fall" and confidence > 0.70:
                state.status = "Alarm"
                alert_img_path = 'alert.jpg'
                cv2.imwrite(alert_img_path, frames[-1]) # Save last frame
                
                # Fire and forget async call to telegram
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(telegram_service.send_alert(alert_img_path), loop)
            else:
                state.status = "Normal"
                
    except Exception as e:
        print(f"Inference Error: {e}")
    finally:
        state.is_inferring = False

def camera_loop():
    global state
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_counter += 1
        
        if frame_counter % 15 == 0:
            state.frame_buffer.append(frame.copy())
            
        if len(state.frame_buffer) == state.buffer_size and not state.is_inferring:
            state.is_inferring = True
            frames_snapshot = list(state.frame_buffer)
            thread = threading.Thread(target=run_inference, args=(frames_snapshot,), daemon=True)
            thread.start()
            
        # Draw HOG Person boxes (lightweight)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Overlay Text
        if state.latest_texts:
            for i, text in enumerate(state.latest_texts):
                y_offset = 40 + (i * 35)
                color = (0, 0, 255) if state.status == "Alarm" else (255, 255, 255)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
        # Status overlay
        status_text = "Status: " + state.status
        color = (0, 0, 255) if state.status == "Alarm" else (0, 255, 0)
        cv2.putText(frame, status_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        state.frame = buffer.tobytes()
        
        time.sleep(0.01) # Small sleep to prevent 100% CPU on this thread

@app.on_event("startup")
async def startup_event():
    global model, loop
    print(f"Loading TimeSformer model on {DEVICE}...")
    try:
        model = init_recognizer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
        print("Model loaded.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    loop = asyncio.get_running_loop()
    
    # Start camera thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

@app.on_event("shutdown")
async def shutdown_event():
    await telegram_service.close()

def video_gen():
    while True:
        if state.frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + state.frame + b'\r\n')
        time.sleep(0.03)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(video_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
def get_status():
    return {"status": state.status}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
