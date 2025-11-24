import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# --- Configuration ---
YOLO_MODEL_PATH = 'models/yolov8/best.pt'
LSTM_MODEL_PATH = 'models/lstm/theft_detector_lstm.h5'
VIDEO_SOURCE = 'data/test_videos/my_test_video.mp4' # Path to your test video
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.9 # Minimum prediction confidence to be considered theft

# --- Feature Extraction Constants (must match feature_extractor.py) ---
MAX_FEATURES_PER_FRAME = 10
FEATURE_VECTOR_SIZE = 6

# --- Load Models ---
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)

def extract_frame_features(frame):
    """Extracts features from a single frame."""
    results = yolo_model(frame, verbose=False)
    frame_features = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()
            feature_vector = [class_id, conf] + xywhn
            frame_features.extend(feature_vector)
            
    # Pad features to fixed size
    if len(frame_features) > MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE:
        frame_features = frame_features[:MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE]
    else:
        padding_needed = MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE - len(frame_features)
        frame_features.extend([0] * padding_needed)
        
    return frame_features

# --- Main Detection Loop ---
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
sequence = deque(maxlen=SEQUENCE_LENGTH)
frame_number = 0
theft_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Extract features for the current frame
    features = extract_frame_features(frame)
    sequence.append(features)

    # 2. Once the sequence is full, make a prediction
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(sequence), axis=0) # Add batch dimension
        prediction = lstm_model.predict(input_data)[0][0]

        if prediction > CONFIDENCE_THRESHOLD:
            theft_detected = True
            start_time = max(0, (frame_number - SEQUENCE_LENGTH) / fps)
            end_time = frame_number / fps
            print(f"!!! THEFT DETECTED !!! Time: {start_time:.2f}s - {end_time:.2f}s | Confidence: {prediction:.2f}")
        else:
            theft_detected = False

    # 3. Display the result on the frame
    display_text = "Status: THEFT DETECTED" if theft_detected else "Status: Normal"
    color = (0, 0, 255) if theft_detected else (0, 255, 0)
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Shoplifting Detection', frame)

    frame_number += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()