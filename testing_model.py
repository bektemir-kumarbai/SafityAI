import cv2
import torch
import numpy as np
import os
import threading
from collections import deque
from mmaction.apis import init_recognizer, inference_recognizer


CHECKPOINT_FILE = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth'
CONFIG_FILE = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEMP_VIDEO_PATH = 'temp_clip.avi'

LABELS_PATH = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\label_map_k400.txt'
with open(LABELS_PATH, 'r') as f:
    LABELS = [line.strip() for line in f.readlines()]

model = init_recognizer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)

buffer_size = 16
frame_buffer = deque(maxlen=buffer_size)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Shared state between main thread and inference thread ---
latest_texts = []
is_inferring = False  # Lock to prevent overlapping inference calls

def run_inference(frames):
    global latest_texts, is_inferring
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(TEMP_VIDEO_PATH, fourcc, 8, (frame_width, frame_height))
        for f in frames:
            writer.write(f)
        writer.release()

        if os.path.exists(TEMP_VIDEO_PATH) and os.path.getsize(TEMP_VIDEO_PATH) > 0:
            result = inference_recognizer(model, TEMP_VIDEO_PATH)
            pred_scores = result.pred_score.cpu().numpy()
            top_k = 4
            top_indices = np.argsort(pred_scores)[-top_k:][::-1]
            new_texts = []
            print("-" * 30)
            for idx in top_indices:
                confidence = pred_scores[idx]
                action_name = LABELS[idx] if idx < len(LABELS) else f"ID:{idx}"
                text_line = f"{action_name}: {confidence * 100:.2f}%"
                new_texts.append(text_line)
                print(f"Action: {action_name} | Confidence: {confidence * 100:.2f}%")
            latest_texts = new_texts
    except Exception as e:
        print(f"Inference Error: {type(e).__name__}: {e}")
    finally:
        is_inferring = False  # ✅ Always unlock when done

print(f"✅ Camera Started at {frame_width}x{frame_height}. Waiting for buffer to fill...")
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1

    if frame_counter % 15 == 0:
        frame_buffer.append(frame.copy())
    if len(frame_buffer) == buffer_size and not is_inferring:
        is_inferring = True
        frames_snapshot = list(frame_buffer)
        thread = threading.Thread(target=run_inference, args=(frames_snapshot,), daemon=True)
        thread.start()
    # ✅ Main thread is never blocked — always shows live feed
    if latest_texts:
        for i, text in enumerate(latest_texts):
            y_offset = 40 + (i * 35)
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('SafetyAI Phase 1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()