# backend/test_processing.py
import os
import tempfile
import time  # Import the time library
from processing import analyze_video_and_get_clips

# --- IMPORTANT ---
# Put the EXACT SAME Cloudinary URL of a SHORT (10-20 seconds) test video here.
# Using a short video will make the test run much faster.
TEST_VIDEO_URL = "https://res.cloudinary.com/djn6ckph6/video/upload/v1729215982/your_video_id.mp4" # <--- REPLACE WITH YOUR SHORT VIDEO URL

def run_test():
    print("--- Starting Direct Processing Test ---")
    start_time = time.time()

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[Checkpoint 1] Created temporary directory: {temp_dir}")
        
        try:
            print(f"[Checkpoint 2] Calling analyze_video_and_get_clips with URL: {TEST_VIDEO_URL}")
            
            # This is the function we are testing
            results = analyze_video_and_get_clips(TEST_VIDEO_URL, temp_dir)
            
            # If the script gets here, the function has successfully completed
            end_time = time.time()
            print("\n--- TEST SUCCEEDED ---")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")

            if results:
                print(f"Successfully processed video and found {len(results)} clips.")
            else:
                print("Processing completed, but no theft events were detected.")
        
        except Exception as e:
            # If any error occurs, it will be caught here
            end_time = time.time()
            print("\n--- !!! TEST FAILED !!! ---")
            print(f"Test ran for {end_time - start_time:.2f} seconds before failing.")
            print("An error occurred during video processing:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_test()