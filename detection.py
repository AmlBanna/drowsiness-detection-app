# detection.py
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time

# ==================== CONFIG ====================
MODEL_PATH = Path("models/improved_cnn_best.keras")
INPUT_SIZE = (48, 48)
CLOSED_THRESHOLD = 5
BASE_SCALE = 0.7
MIN_FACE_SIZE = 80
FRAME_SKIP_MAX = 2
# ===============================================

# Load Model
model = tf.keras.models.load_model(str(MODEL_PATH))

def predict_batch(eyes):
    if len(eyes) == 0: return np.array([])
    return model.predict(eyes, verbose=0).flatten()

# DNN Face Detector
net = None
try:
    net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
except:
    pass

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def preprocess_eye(eye_img):
    eye = cv2.resize(eye_img, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    eye = eye.astype(np.float32) / 255.0
    return np.expand_dims(eye, axis=-1)

def detect_frame(frame, closed_counter=0, skip_counter=0, last_state='open', fps_start=0, frame_count=0):
    h_orig, w_orig = frame.shape[:2]
    scale = BASE_SCALE if min(w_orig, h_orig) >= 400 else 1.0
    small_frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Frame skipping
    if last_state == 'open' and closed_counter == 0:
        skip_counter += 1
        if skip_counter <= FRAME_SKIP_MAX:
            return frame, closed_counter, skip_counter, last_state, fps_start, frame_count, False
    skip_counter = 0

    # FPS
    frame_count += 1
    if frame_count % 20 == 0:
        fps = 20 / (time.time() - fps_start + 1e-6)
        fps_start = time.time()
    else:
        fps = 0

    eyes_batch = []
    eye_boxes = []
    eyes_closed = False
    eyes_detected = False
    faces = []

    # Face Detection
    if net:
        try:
            h, w = small_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(small_frame, 1.0, (300, 300), (104, 177, 123))
            net.setInput(blob)
            detections = net.forward()
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf < 0.5: continue
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                fw, fh = x2-x1, y2-y1
                if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE: continue
                faces.append((x1, y1, fw, fh))
        except:
            net = None

    if not faces:
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE))
        faces = [(x, y, w, h) for x, y, w, h in faces]

    # Eye Detection
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+int(h*0.65), x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.05, 4, minSize=(20,20), maxSize=(80,80))
        for (ex, ey, ew, eh) in eyes:
            if ey > roi_gray.shape[0] * 0.55: continue
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_img.size == 0 or min(ew, eh) < 18: continue
            eyes_detected = True
            eyes_batch.append(preprocess_eye(eye_img))
            sx = w_orig / small_frame.shape[1]
            sy = h_orig / small_frame.shape[0]
            ex_full = int((x + ex) * sx)
            ey_full = int((y + ey) * sy)
            ew_full = int(ew * sx)
            eh_full = int(eh * sy)
            eye_boxes.append((ex_full, ey_full, ew_full, eh_full))

    preds = predict_batch(np.array(eyes_batch)) if eyes_batch else np.array([])
    current_state = 'unknown'
    for i, (pred, box) in enumerate(zip(preds, eye_boxes)):
        is_open = pred > 0.5
        conf = pred if is_open else 1 - pred
        color = (0, 255, 0) if is_open else (0, 0, 255)
        label = f"{'OPEN' if is_open else 'CLOSED'} {conf:.2f}"
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
        cv2.putText(frame, label, (box[0], box[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if not is_open:
            eyes_closed = True
            current_state = 'closed'
        else:
            current_state = 'open'

    if not eyes_detected and faces:
        cv2.putText(frame, "Eyes not visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    # Drowsiness
    if eyes_closed:
        closed_counter += 1
    elif eyes_detected:
        closed_counter = max(0, closed_counter - 1)

    alert = closed_counter >= CLOSED_THRESHOLD
    if alert:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (0, 0, 255), -1)
        cv2.putText(frame, "DROWSINESS ALERT!", (60, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.putText(frame, "WAKE UP!", (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    status = "CLOSED" if eyes_closed else "OPEN"
    color = (0, 0, 255) if eyes_closed else (0, 255, 0)
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if closed_counter > 0:
        cv2.putText(frame, f"Closed: {closed_counter}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    return frame, closed_counter, skip_counter, current_state, fps_start, frame_count, alert
