# backend/processing.py

import cv2
import numpy as np
import os
import time
from collections import deque
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import google.generativeai as genai

# --- Configuration ---
YOLO_MODEL_PATH = 'models/best.pt'
LSTM_MODEL_PATH = 'models/lstm/theft_detector_lstm.h5'
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.85  # Slightly lower to catch more possible thefts for Gemini to review

# --- Feature Extraction Constants ---
MAX_FEATURES_PER_FRAME = 10
FEATURE_VECTOR_SIZE = 6

# --- Google AI Gemini Configuration ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
else:
    print("[CRITICAL WARNING] GOOGLE_API_KEY not found. Gemini verification will be skipped.")
    gemini_model = None

# --- Load Models ---
print("Loading YOLOv8 model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("Loading LSTM model...")
lstm_model = load_model(LSTM_MODEL_PATH)
print("Models loaded successfully.")


# ==============================
# Helper Functions
# ==============================

def extract_frame_features(frame):
    """Extracts a fixed-length feature vector from a single frame using YOLOv8."""
    results = yolo_model(frame, verbose=False)
    frame_features = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            xywhn = box.xywhn[0].tolist()
            frame_features.extend([class_id, conf] + xywhn)
    padding_needed = (MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE) - len(frame_features)
    if padding_needed > 0:
        frame_features.extend([0] * padding_needed)
    else:
        frame_features = frame_features[:MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE]
    return frame_features


def merge_timestamps(timeframes, buffer_seconds=1.0):
    """Merges overlapping or continuous timeframes into single events."""
    if not timeframes:
        return []
    sorted_frames = sorted(timeframes, key=lambda x: x[0])
    merged = [list(sorted_frames[0])]
    for current_start, current_end in sorted_frames[1:]:
        last_start, last_end = merged[-1]
        if current_start <= (last_end + buffer_seconds):
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])
    return [tuple(event) for event in merged]


def create_clip_with_boxes(original_video_path, output_path, start_time, end_time, fps):
    """Creates an annotated clip for a given time range with YOLO bounding boxes."""
    cap = cv2.VideoCapture(original_video_path)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use H.264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"        -> [ERROR] Failed to open VideoWriter for {output_path}")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_num = start_frame

    print(f"        -> Annotating frames {start_frame} to {end_frame}...")
    while cap.isOpened() and current_frame_num <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        out.write(frame)
        current_frame_num += 1

    cap.release()
    out.release()
    print(f"        -> Finished writing annotated clip to {output_path}")


# ==============================
# Gemini Verification Function
# ==============================

def verify_clip_with_gemini(clip_path: str) -> bool:
    """Uploads a video clip and asks Gemini if it shows theft. Returns True/False."""
    if not gemini_model:
        print("    [GEMINI] Skipping verification (Gemini not configured).")
        return True  # Default True when Gemini is off (to not block detections)

    print(f"    [GEMINI] Uploading '{os.path.basename(clip_path)}' for verification...")

    try:
        video_file = genai.upload_file(path=clip_path, mime_type="video/mp4")
        while video_file.state.name == "PROCESSING":
            print('.', end='', flush=True)
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
    except Exception as e:
        print(f"\n    [GEMINI] ERROR uploading file: {e}")
        return False

    if video_file.state.name == "FAILED":
        print(f"\n    [GEMINI] File processing failed for {clip_path}")
        return False

    print("\n    [GEMINI] Asking Gemini for visual analysis...")
    prompt = (
        "You are a professional loss prevention expert. "
        "Analyze this short security video. "
        "Does this video show a person intentionally concealing an item in their pocket, jacket, or personal bag? "
        "Answer strictly with 'YES' or 'NO'."
    )

    try:
        response = gemini_model.generate_content([prompt, video_file])
        verdict = response.text.strip().upper()
        print(f"    [GEMINI] Verdict: {verdict}")
        genai.delete_file(video_file.name)
        return "YES" in verdict
    except Exception as e:
        print(f"    [GEMINI] API ERROR: {e}")
        return False


# ==============================
# Main Analysis Function
# ==============================

def analyze_video_and_get_clips(video_path: str, output_dir: str):
    """
    Analyzes a video file, detects potential theft events using LSTM,
    generates annotated clips with YOLO, and verifies them using Gemini.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: Could not open video file: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
        print("    WARNING: Could not determine FPS. Defaulting to 30.")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = total_frames / fps if fps > 0 else 0

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    frame_number = 0
    detected_timeframes = []

    print("    [AI-TASK] Phase 1: Running YOLO+LSTM to detect suspicious timestamps...")
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
        frame_number += 1
    cap.release()

    print("    [AI-TASK] Phase 1 finished.")
    merged_events = merge_timestamps(detected_timeframes)
    print(f"    [AI-TASK] Found {len(merged_events)} distinct events: {merged_events}")

    if not merged_events:
        print("    [RESULT] No suspicious activity detected.")
        return []

    verified_clips_info = []
    print("    [AI-TASK] Phase 2: Generating annotated clips and verifying with Gemini...")

    for i, (start_time, end_time) in enumerate(merged_events):
        print(f"\n--- Processing Event #{i+1} ---")
        if end_time <= start_time:
            print(f"    [WARNING] Skipping event #{i+1} due to invalid duration.")
            continue

        output_filename = os.path.join(output_dir, f"clip_{i}.mp4")
        clip_start = max(0, start_time - 1)
        clip_end = min(video_duration, end_time + 1)

        print(f"    -> Creating annotated clip for {clip_start:.2f}s–{clip_end:.2f}s")
        create_clip_with_boxes(video_path, output_filename, clip_start, clip_end, fps)

        # Phase 3: Verify with Gemini
        is_theft = verify_clip_with_gemini(output_filename)
        if is_theft:
            print(f"    [VERIFIED] Event #{i+1} confirmed as theft.")
            try:
                summary = (
                    f"A confirmed theft event was detected between "
                    f"{start_time:.1f}s and {end_time:.1f}s, verified by Gemini."
                )
                verified_clips_info.append({
                    "start": start_time,
                    "end": end_time,
                    "local_path": output_filename,
                    "summary": summary
                })
            except FileNotFoundError:
                print(f"        -> [ERROR] File not found: {output_filename}")
        else:
            print(f"    [REJECTED] Event #{i+1} dismissed by Gemini as false positive.")
            if os.path.exists(output_filename):
                os.remove(output_filename)

    print("\n    [AI-TASK] All potential events processed.")
    return verified_clips_info
