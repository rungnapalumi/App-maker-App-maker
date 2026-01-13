import os
import json
import boto3
import cv2
import numpy as np
from datetime import datetime
from mediapipe.python.solutions import pose as mp_pose

# Load environment variables
AWS_BUCKET = os.environ["AWS_BUCKET"]
AWS_REGION = os.environ["AWS_REGION"]
JOB_POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL", 10))

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
OUTPUT_PREFIX = "jobs/output/"
FAILED_PREFIX = "jobs/failed/"

def read_job_json(key):
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read())

def write_json_to_s3(data, key):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json"
    )

def process_dot_video(input_path, output_path, dot_size=2):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            black = np.zeros((h, w, 3), dtype=np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            black, 
                            (cx, cy), 
                            dot_size, 
                            (255, 255, 255), 
                            -1
                        )

            out.write(black)

    cap.release()
    out.release()

def process_job(job_id):
    json_key = f"{PENDING_PREFIX}{job_id}.json"
    folder = f"{PENDING_PREFIX}{job_id}/"
    video_key = f"{folder}input/input.mp4"

    local_in = "/tmp/input.mp4"
    local_out = "/tmp/output.mp4"

    print(f"[JOB] Start job {job_id}")

    try:
        s3.download_file(AWS_BUCKET, video_key, local_in)
    except Exception as e:
        print("[ERROR] Cannot download input video:", e)
        write_json_to_s3(
            {"job_id": job_id, "status": "failed", "error": "Cannot download video"},
            f"{FAILED_PREFIX}{job_id}.json"
        )
        return

    try:
        process_dot_video(local_in, local_out)
    except Exception as e:
        print("[ERROR] Video processing failed:", e)
        write_json_to_s3(
            {"job_id": job_id, "status": "failed", "error": str(e)},
            f"{FAILED_PREFIX}{job_id}.json"
        )
        return

    out_key = f"{OUTPUT_PREFIX}{job_id}.mp4"
    json_out_key = f"{OUTPUT_PREFIX}{job_id}.json"

    try:
        s3.upload_file(local_out, AWS_BUCKET, out_key)

        write_json_to_s3(
            {"job_id": job_id, "status": "done", "video": out_key},
            json_out_key
        )
        print(f"[DONE] Job {job_id} finished")

    except Exception as e:
        print("[ERROR] Failed uploading output:", e)
        write_json_to_s3(
            {"job_id": job_id, "status": "failed", "error": "Upload failed"},
            f"{FAILED_PREFIX}{job_id}.json"
        )

def poll_jobs():
    print("Worker started. Waiting for jobs...")
    while True:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
        items = resp.get("Contents", [])

        jobs = [x["Key"] for x in items if x["Key"].endswith(".json")]

        if jobs:
            job_key = jobs[0]
            job_id = job_key.split("/")[-1].replace(".json", "")
            process_job(job_id)

        else:
            print("No pending jobs. Sleeping...")
        
        import time
        time.sleep(JOB_POLL_INTERVAL)

if __name__ == "__main__":
    poll_jobs()
