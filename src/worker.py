# src/worker.py
import os
import time
import json
import shutil
import tempfile
import boto3
import cv2
import numpy as np
from datetime import datetime

# =====================
# CONFIG
# =====================
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET")

DOT_RADIUS = 5
POLL_INTERVAL = 5  # seconds

s3 = boto3.client("s3", region_name=AWS_REGION)

# =====================
# UTIL
# =====================
def log(msg):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

def s3_list(prefix):
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    return resp.get("Contents", [])

def s3_move(src_key, dst_key):
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)

# =====================
# VIDEO PROCESS (Johansson DOTS)
# =====================
def render_johansson_dots(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    prev_gray = None
    dots = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # init random dots
            dots = np.random.randint(0, [w, h], (150, 2))
            prev_gray = gray
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            for i in range(len(dots)):
                x, y = dots[i]
                dx, dy = flow[y % h, x % w]
                dots[i] = [
                    int(np.clip(x + dx, 0, w - 1)),
                    int(np.clip(y + dy, 0, h - 1)),
                ]
            prev_gray = gray

        canvas = np.zeros_like(frame)
        for x, y in dots:
            cv2.circle(canvas, (int(x), int(y)), DOT_RADIUS, (255, 255, 255), -1)

        out.write(canvas)

    cap.release()
    out.release()

# =====================
# JOB HANDLER
# =====================
def handle_job(job_key):
    job_id = job_key.split("/")[-1].replace(".json", "")
    log(f"Processing job {job_id}")

    with tempfile.TemporaryDirectory() as tmp:
        job_json = os.path.join(tmp, "job.json")
        s3.download_file(S3_BUCKET, job_key, job_json)

        with open(job_json) as f:
            job = json.load(f)

        input_key = job["input_key"]
        local_input = os.path.join(tmp, "input.mp4")
        local_output = os.path.join(tmp, "output.mp4")

        s3.download_file(S3_BUCKET, input_key, local_input)

        render_johansson_dots(local_input, local_output)

        output_key = f"jobs/{job_id}/output/dots.mp4"
        s3.upload_file(local_output, S3_BUCKET, output_key)

        job["status"] = "done"
        job["outputs"] = {"overlay_key": output_key}
        job["updated_at"] = datetime.utcnow().isoformat()

        done_key = f"jobs/done/{job_id}.json"
        with open(job_json, "w") as f:
            json.dump(job, f)

        s3.upload_file(job_json, S3_BUCKET, done_key)

# =====================
# MAIN LOOP
# =====================
def main():
    log("Worker boot (Johansson DOTS ONLY)")
    log(f"AWS_REGION={AWS_REGION}")
    log(f"S3_BUCKET={S3_BUCKET}")

    while True:
        pending = s3_list("jobs/pending/")
        if not pending:
            time.sleep(POLL_INTERVAL)
            continue

        job_obj = pending[0]
        job_key = job_obj["Key"]

        processing_key = job_key.replace("jobs/pending/", "jobs/processing/")
        s3_move(job_key, processing_key)

        try:
            handle_job(processing_key)
            s3_move(processing_key, processing_key.replace("processing", "done"))
        except Exception as e:
            log(f"Job failed: {e}")
            s3_move(processing_key, processing_key.replace("processing", "failed"))

# =====================
if __name__ == "__main__":
    main()
