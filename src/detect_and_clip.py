import cv2
import numpy as np
import os
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from moviepy.editor import VideoFileClip

# --- Configuration ---
YOLO_MODEL_PATH = 'models/yolov8/best.pt'
LSTM_MODEL_PATH = 'models/lstm/theft_detector_lstm.h5'
VIDEO_SOURCE = 'data/test_videos/text1.mp4' # Path to your main video
OUTPUT_CLIPS_DIR = 'output_clips/' # Folder to save the theft clips

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.90

# --- Feature Extraction Constants (must match feature_extractor.py) ---
MAX_FEATURES_PER_FRAME = 10
FEATURE_VECTOR_SIZE = 6

# --- Load Models ---
print("Loading models...")
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = load_model(LSTM_MODEL_PATH)
print("Models loaded successfully.")

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

def merge_timestamps(timeframes):
    """Merges overlapping timeframes into single events."""
    if not timeframes:
        return []

    # Sort by start time
    sorted_frames = sorted(timeframes, key=lambda x: x[0])
    
    merged = [sorted_frames[0]]
    for current_start, current_end in sorted_frames[1:]:
        last_start, last_end = merged[-1]
        
        # If the current frame overlaps or is continuous, merge it
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
            
    return merged

def main():
    # --- Phase 1: Detect all theft events and collect timestamps ---
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video file at path: {VIDEO_SOURCE}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    frame_number = 0
    detected_timeframes = []

    print("Processing video to detect events...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_frame_features(frame)
        sequence.append(features)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(sequence), axis=0)
            prediction = lstm_model.predict(input_data, verbose=0)[0][0]

            if prediction > CONFIDENCE_THRESHOLD:
                start_time = max(0, (frame_number - SEQUENCE_LENGTH) / fps)
                end_time = frame_number / fps
                detected_timeframes.append((start_time, end_time))
                print(f"Potential theft detected at {start_time:.2f}s - {end_time:.2f}s")
        
        frame_number += 1
    
    cap.release()
    print("Video processing finished.")

    # --- Phase 2: Merge overlapping timestamps ---
    if not detected_timeframes:
        print("No theft events were detected.")
        return

    print("Merging detected timeframes into distinct events...")
    merged_events = merge_timestamps(detected_timeframes)
    print(f"Found {len(merged_events)} distinct theft event(s).")
    for i, (start, end) in enumerate(merged_events):
        print(f"  Event {i+1}: from {start:.2f}s to {end:.2f}s")

    # --- Phase 3: Clip the video and save the events ---
    print("Clipping events from the main video...")
    if not os.path.exists(OUTPUT_CLIPS_DIR):
        os.makedirs(OUTPUT_CLIPS_DIR)

    with VideoFileClip(VIDEO_SOURCE) as video:
        for i, (start_time, end_time) in enumerate(merged_events):
            event_num = i + 1
            output_filename = os.path.join(OUTPUT_CLIPS_DIR, f"theft_event_{event_num}.mp4")
            
            print(f"  Clipping event {event_num} and saving to {output_filename}...")
            # Add a small buffer to the clip to see context, e.g., 1 second before and after
            clip_start = max(0, start_time - 1)
            clip_end = min(video.duration, end_time + 1)
            
            new_clip = video.subclip(clip_start, clip_end)
            new_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    print("All detected events have been clipped and saved.")

if __name__ == '__main__':
    main()