import os
import tempfile
import requests
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, auth

from processing import analyze_video_and_get_clips

# --- Initialization ---
load_dotenv()
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost:5173", # The address of your React frontend
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Connections ---
try:
    print("Connecting to MongoDB...")
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client.theft_detection_db
    print("MongoDB connection successful.")

    print("Initializing Firebase Admin SDK...")
    cred = credentials.Certificate(os.getenv("FIREBASE_ADMIN_SDK_PATH"))
    firebase_admin.initialize_app(cred)
    print("Firebase Admin SDK initialized successfully.")

    print("Configuring Cloudinary...")
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET")
    )
    print("Cloudinary configured successfully.")
except Exception as e:
    print(f"FATAL: An error occurred during initialization: {e}")

# --- Pydantic Models ---
class VideoProcessRequest(BaseModel):
    videoUrl: str
    fileName: str

# --- Background Worker ---
def video_processing_worker(job_id_str: str, video_url: str, user_id: str):
    job_id = ObjectId(job_id_str)
    print(f"[WORKER] Starting job {job_id_str} for user {user_id}.")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"[WORKER] Job {job_id_str}: Created temp directory at {temp_dir}.")
            
            local_video_path = os.path.join(temp_dir, "original_video.mp4")
            print(f"[WORKER] Job {job_id_str}: Downloading video from {video_url}...")
            
            with requests.get(video_url, stream=True) as r:
                r.raise_for_status()
                with open(local_video_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print(f"[WORKER] Job {job_id_str}: Video downloaded. Starting AI analysis...")
            results = analyze_video_and_get_clips(local_video_path, temp_dir)
            print(f"[WORKER] Job {job_id_str}: AI analysis complete. Found {len(results)} clips.")
            
            final_results = []
            if results:
                print(f"[WORKER] Job {job_id_str}: Uploading result clips to Cloudinary...")
                for result in results:
                    cloud_response = cloudinary.uploader.upload(
                        result["local_path"],
                        resource_type="video",
                        folder=f"clips/{user_id}/{job_id_str}"
                    )
                    final_results.append({
                        "start": result["start"],
                        "end": result["end"],
                        "url": cloud_response['secure_url'],
                        "summary": result["summary"]
                    })
                print(f"[WORKER] Job {job_id_str}: Clip upload complete.")

        db.jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": "completed", "results": final_results}}
        )
        print(f"[WORKER] Job {job_id_str} completed and DB updated.")

    except Exception as e:
        print(f"\n--- [WORKER] CRITICAL ERROR in Job {job_id_str} ---")
        traceback.print_exc()
        db.jobs.update_one(
            {"_id": job_id},
            {"$set": {"status": "failed", "error": str(e)}}
        )
        print(f"[WORK-ER] Job {job_id_str} marked as FAILED in DB.\n")

# --- API Endpoints ---
@app.post("/process", status_code=202)
async def process_video(request: VideoProcessRequest, background_tasks: BackgroundTasks, http_request: Request):
    try:
        token = http_request.headers.get("Authorization").split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
    except (IndexError, AttributeError, auth.InvalidIdTokenError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {e}")

    job = db.jobs.insert_one({
        "userId": user_id,
        "originalUrl": request.videoUrl,
        "fileName": request.fileName,
        "status": "processing",
        "createdAt": datetime.now(timezone.utc),
        "results": [],
        "error": None
    })
    job_id_str = str(job.inserted_id)
    background_tasks.add_task(video_processing_worker, job_id_str, request.videoUrl, user_id)
    return {"message": "Processing started", "jobId": job_id_str}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    try:
        job = db.jobs.find_one({"_id": ObjectId(job_id)})
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job["_id"] = str(job["_id"])
        if 'createdAt' in job and job['createdAt']:
             job['createdAt'] = job['createdAt'].isoformat()
        return job
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Job ID format")

@app.get("/history")
async def get_history(http_request: Request):
    try:
        token = http_request.headers.get("Authorization").split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
    except (IndexError, AttributeError, auth.InvalidIdTokenError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {e}")

    user_jobs = list(db.jobs.find({"userId": user_id}).sort("createdAt", -1))
    
    for job in user_jobs:
        job["_id"] = str(job["_id"])
        if 'createdAt' in job and job['createdAt']:
             job['createdAt'] = job['createdAt'].isoformat()
    return user_jobs

@app.delete("/history/{job_id}")
async def delete_history_item(job_id: str, http_request: Request):
    try:
        token = http_request.headers.get("Authorization").split("Bearer ")[1]
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token['uid']
    except (IndexError, AttributeError, auth.InvalidIdTokenError) as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {e}")

    delete_result = db.jobs.find_one_and_delete({
        "_id": ObjectId(job_id),
        "userId": user_id
    })

    if delete_result is None:
        raise HTTPException(status_code=404, detail="Job not found or you don't have permission to delete it.")
    
    return {"message": "Job deleted successfully"}