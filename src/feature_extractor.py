import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
YOLO_MODEL_PATH = 'models/yolov8/best.pt'
RAW_VIDEOS_DIR = 'data/raw_videos/'
FEATURES_DIR = 'data/processed_features/'
MAX_FEATURES_PER_FRAME = 10 # Max number of detected objects to consider in a frame
FEATURE_VECTOR_SIZE = 6 # e.g., class_id, conf, x_center, y_center, width, height

# --- Load YOLO Model ---
yolo_model = YOLO(YOLO_MODEL_PATH)

def extract_features(video_path):
    """Extracts features from a single video file."""
    cap = cv2.VideoCapture(video_path)
    video_features = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        frame_features = []
        
        # Extract features for each detected box
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                xywhn = box.xywhn[0].tolist() # Normalized [x, y, w, h]
                
                # We create a fixed-size vector for each detection
                feature_vector = [class_id, conf] + xywhn
                frame_features.extend(feature_vector)
        
        # Pad the features for the current frame to a fixed size
        if len(frame_features) > MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE:
            frame_features = frame_features[:MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE]
        else:
            padding_needed = MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE - len(frame_features)
            frame_features.extend([0] * padding_needed)
            
        video_features.append(frame_features)

    cap.release()
    return np.array(video_features)

def process_all_videos():
    """Processes all videos in the raw_videos directory."""
    for category in ['theft', 'normal']:
        category_path = os.path.join(RAW_VIDEOS_DIR, category)
        output_category_path = os.path.join(FEATURES_DIR, category)
        
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        videos = [v for v in os.listdir(category_path) if v.endswith('.mp4')]
        
        for video_name in tqdm(videos, desc=f'Processing {category} videos'):
            video_path = os.path.join(category_path, video_name)
            features = extract_features(video_path)
            
            output_filename = os.path.splitext(video_name)[0] + '.npy'
            output_path = os.path.join(output_category_path, output_filename)
            np.save(output_path, features)
            print(f"Saved features for {video_name} to {output_path}")

if __name__ == '__main__':
    process_all_videos()