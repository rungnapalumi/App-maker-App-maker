import os
import time
import json
import uuid
import cv2
import boto3
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
DOT_RADIUS = 5
MAX_CORNERS = 200
QUALITY_LEVEL = 0.01
MIN_DISTANCE = 7
BLOCK_SIZE = 7

S3_BUCKET = os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

# =========================
# S3 CLIENT (SAFE)
# =========================
s3 = boto3.client("s3", region_name=AWS_REGION)

print("✅ Worker boot (Optical Flow Johansson DOTS ONLY)")
print("AWS_REGION =", AWS_REGION)
print("S3_BUCKET =", S3_BUCKET)

# =========================
# UTILS
# =========================
def now():
    return datetime.utcnow().isoformat() + "Z"


def safe_xy(pt):
    """
    Normalize optical-flow point to (x, y)
    Handles: [x,y], [[x,y]], [x,y,dx,dy], etc.
    """
    arr = np.array(pt).reshape(-1)
    if len(arr) < 2:
        return None
    return int(arr[0]), int(arr[1])


# =========================
# CORE: JOHANSSON DOTS
# =========================
def render_johansson_dots_optical_flow(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
    )

    if p0 is None:
        p0 = np.empty((0, 1, 2), dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None
            )

            good_new = p1[st == 1] if p1 is not None else []
        else:
            good_new = []

        # Black background
        canvas = np.zeros_like(frame)

        for pt in good_new:
            xy = safe_xy(pt)
            if xy is None:
                continue
            x, y = xy
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (x, y), DOT_RADIUS, (255, 255, 255), -1)

        out.write(canvas)

        prev_gray = gray
        p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else np.empty((0, 1, 2), dtype=np.float32)

    cap.release()
    out.release()


# =========================
# JOB HANDLER
# =========================
def handle_job(job):
    job_id = job["job_id"]
    input_key = f"jobs/{job_id}/input/input.mp4"
    output_key = f"jobs/{job_id}/output/overlay.mp4"

    local_in = f"/tmp/{job_id}_in.mp4"
    local_out = f"/tmp/{job_id}_out.mp4"

    print("⬇️ download", input_key)
    s3.download_file(S3_BUCKET, input_key, local_in)

    print("🎥 Rendering Johansson dots (optical flow)")
    render_johansson_dots_optical_flow(local_in, local_out)

    print("⬆️ upload", output_key)
    s3.upload_file(local_out, S3_BUCKET, output_key)

    job["status"] = "done"
    job["progress"] = 100
    job["outputs"] = {"overlay_key": output_key}
    job["updated_at"] = now()

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"jobs/{job_id}/status.json",
        Body=json.dumps(job).encode("utf-8"),
        ContentType="application/json",
    )

    print("✅ Job done:", job_id)


# =========================
# MAIN LOOP
# =========================
def main():
    while True:
        # Poll jobs
        resp = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix="jobs/",
            Delimiter="/",
        )

        prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]

        for p in prefixes:
            status_key = p + "status.json"
            try:
                obj = s3.get_object(Bucket=S3_BUCKET, Key=status_key)
                job = json.loads(obj["Body"].read())
            except:
                continue

            if job.get("status") == "queued":
                print("▶️ Processing job:", job["job_id"])
                job["status"] = "processing"
                job["updated_at"] = now()

                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=status_key,
                    Body=json.dumps(job).encode("utf-8"),
                    ContentType="application/json",
                )

                try:
                    handle_job(job)
                except Exception as e:
                    print("❌ Job failed:", job["job_id"], e)
                    job["status"] = "failed"
                    job["message"] = str(e)
                    job["updated_at"] = now()
                    s3.put_object(
                        Bucket=S3_BUCKET,
                        Key=status_key,
                        Body=json.dumps(job).encode("utf-8"),
                        ContentType="application/json",
                    )

        time.sleep(3)


if __name__ == "__main__":
    main()
