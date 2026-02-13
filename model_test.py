import cv2
import torch
import numpy as np
from collections import deque
from mmaction.apis import init_recognizer, inference_recognizer

CHECKPOINT_FILE  = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb_20220815-a4d0d01f.pth'
CONFIG_FILE = r'C:\Users\bahti\PycharmProjects\SafityAI\checkpoints\timesformer_divST_8xb8-8x32x1-15e_kinetics400-rgb.py'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = init_recognizer(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
buffer_size = 8
frame_buffer = deque(maxlen=buffer_size)
cap = cv2.VideoCapture(0)

print("✅ Camera Started. Waiting for buffer to fill...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Fill Buffer
    frame_buffer.append(frame)

    # 2. Run AI (Only when buffer is full)
    if len(frame_buffer) == buffer_size:
        video_clip = list(frame_buffer)

        try:
            # Run Inference
            result = inference_recognizer(model, video_clip)

            # Get Top Prediction
            pred_scores = result.pred_scores.item().cpu().numpy()
            top_idx = np.argmax(pred_scores)
            confidence = pred_scores[top_idx]

            # --- DISPLAY ---
            # Print to console
            print(f"Action ID: {top_idx} | Confidence: {confidence:.2f}")

            # Draw on screen
            text = f"Action: {top_idx} ({confidence:.2f})"
            color = (0, 255, 0)  # Green

            # Simple Alarm Logic (Example IDs for Fighting/Falling)
            # We will refine these IDs in the next step
            if confidence > 0.5:
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        except Exception as e:
            print(f"Inference Error: {e}")

            # Show the video feed
    cv2.imshow('SafetyAI Phase 1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()