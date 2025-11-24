# backend/1_run_feature_extractor.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration ---
# Make sure these paths are correct for your project structure
YOLO_MODEL_PATH = 'models/best.pt' 
RAW_VIDEOS_DIR = 'data/raw_videos/'
FEATURES_DIR = 'data/processed_features/'
MAX_FEATURES_PER_FRAME = 10 
FEATURE_VECTOR_SIZE = 6 

# --- Load YOLO Model ---
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO model loaded.")

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    video_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model(frame, verbose=False)
        frame_features = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                xywhn = box.xywhn[0].tolist()
                feature_vector = [class_id, conf] + xywhn
                frame_features.extend(feature_vector)
        
        if len(frame_features) > MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE:
            frame_features = frame_features[:MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE]
        else:
            padding_needed = MAX_FEATURES_PER_FRAME * FEATURE_VECTOR_SIZE - len(frame_features)
            frame_features.extend([0] * padding_needed)
            
        video_features.append(frame_features)
    cap.release()
    return np.array(video_features)

def process_all_videos():
    for category in ['theft', 'normal']:
        category_path = os.path.join(RAW_VIDEOS_DIR, category)
        output_category_path = os.path.join(FEATURES_DIR, category)
        
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        videos = [v for v in os.listdir(category_path) if v.endswith(('.mp4', '.avi', '.mov'))]
        
        for video_name in tqdm(videos, desc=f'Processing {category} videos'):
            video_path = os.path.join(category_path, video_name)
            features = extract_features(video_path)
            
            output_filename = os.path.splitext(video_name)[0] + '.npy'
            output_path = os.path.join(output_category_path, output_filename)
            np.save(output_path, features)

if __name__ == '__main__':
    process_all_videos()
    print("\nFeature extraction complete for all videos!")
    