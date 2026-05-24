import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import torch
import numpy as np
import os
from mmaction.apis import init_recognizer, inference_recognizer

CHECKPOINT_FILE = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth'
CONFIG_FILE = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LABELS_PATH = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\label_map_k400.txt'
with open(LABELS_PATH, 'r', encoding='utf-8') as f:
    LABELS = [line.strip() for line in f.readlines()]

LOG_FILE = 'test_run.txt'
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
    except Exception:
        pass

def log_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs, flush=True)
    except Exception:
        pass

log_print("Initializing TimeSformer model...")
model = init_recognizer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
log_print("Model loaded successfully!")

video_dir = 'videos'
video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

if not video_files:
    log_print("No videos found in the 'videos' folder.")
else:
    for video_path in video_files:
        log_print("\n" + "=" * 50)
        log_print(f"[TEST] Testing video: {video_path}")
        
        # 1. Run inference on the entire video
        try:
            with torch.no_grad():
                result = inference_recognizer(model, video_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract scores
            if hasattr(result, 'pred_score'):
                pred_scores = result.pred_score.cpu().numpy()
            elif hasattr(result, 'pred_scores'):
                pred_scores = result.pred_scores.item.cpu().numpy()
            else:
                pred_scores = result.pred_score.cpu().numpy()
                
            top_k = 4
            top_indices = np.argsort(pred_scores)[-top_k:][::-1]
            
            latest_texts = []
            log_print("Top predictions:")
            for idx in top_indices:
                confidence = pred_scores[idx]
                action_name = LABELS[idx] if idx < len(LABELS) else f"ID:{idx}"
                text_line = f"{action_name}: {confidence * 100:.2f}%"
                latest_texts.append(text_line)
                log_print(f"  - {action_name}: {confidence * 100:.2f}%")
                
        except Exception as e:
            log_print(f"Inference Error on {video_path}: {type(e).__name__}: {e}")
            continue
            
        # 2. Play the video and display the prediction results overlaid
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30
        delay = max(1, int(1000 / fps))
        
        log_print(f"Playing video: {video_path} at {fps} FPS...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to standard size for visualization
            frame = cv2.resize(frame, (640, 480))
            
            # Draw predictions text overlay
            for i, text in enumerate(latest_texts):
                y_offset = 40 + (i * 35)
                # Black background outline for readability
                cv2.putText(frame, text, (12, y_offset + 2), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
                # White text
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                
            cv2.imshow('SafetyAI Phase 1 - Video Tester', frame)
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                log_print("Testing interrupted by user.")
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
                
        cap.release()
        log_print(f"Finished playing: {video_path}")

cv2.destroyAllWindows()
log_print("\nAll videos have been successfully tested!")