# worker.py
# Johansson DOTS ONLY – Stable, Fixed Landmarks, No Blink
# Author: Lumi (for Rung 💙)

import os
import json
import time
import boto3
import cv2
import numpy as np

# --- Optional MediaPipe (safe import) ---
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = hasattr(mp, "solutions")
except Exception:
    mp = None
    HAS_MEDIAPIPE = False

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

# --- Johansson landmark set (15 points) ---
JOHANSSON_LANDMARKS = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# MediaPipe index map
MP_IDX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

# --- Smoothing ---
def ema(prev, curr, alpha=0.7):
    if prev is None:
        return curr
    return alpha * prev + (1 - alpha) * curr

# --- Main render ---
def render_johansson_dots(
    input_video,
    output_video,
    dot_radius=5
):
    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    pose = None
    if HAS_MEDIAPIPE:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    prev_points = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        black = np.zeros_like(frame)

        points = {}

        if pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks:
                for name in JOHANSSON_LANDMARKS:
                    idx = MP_IDX[name]
                    lm = res.pose_landmarks.landmark[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points[name] = np.array([x, y], dtype=np.float32)

        # fallback = keep previous (no blink)
        for k in JOHANSSON_LANDMARKS:
            if k in points:
                points[k] = ema(prev_points.get(k), points[k])
            else:
                if k in prev_points:
                    points[k] = prev_points[k]

        # draw dots
        for p in points.values():
            cv2.circle(
                black,
                (int(p[0]), int(p[1])),
                dot_radius,
                (255, 255, 255),
                -1
            )

        prev_points = points
        out.write(black)

    cap.release()
    out.release()

# --- Worker entry ---
def process_job(job_id, dot_radius):
    base = f"jobs/{job_id}"
    input_key = f"{base}/input/input.mp4"
    output_key = f"{base}/output/dots.mp4"

    os.makedirs("/tmp/input", exist_ok=True)
    os.makedirs("/tmp/output", exist_ok=True)

    local_in = "/tmp/input/input.mp4"
    local_out = "/tmp/output/dots.mp4"

    s3.download_file(S3_BUCKET, input_key, local_in)

    render_johansson_dots(
        local_in,
        local_out,
        dot_radius=dot_radius
    )

    s3.upload_file(local_out, S3_BUCKET, output_key)

    status = {
        "job_id": job_id,
        "status": "done",
        "progress": 100,
        "message": "Completed",
        "outputs": {
            "overlay_key": output_key
        }
    }

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=f"{base}/status.json",
        Body=json.dumps(status),
        ContentType="application/json"
    )

# --- CLI ---
if __name__ == "__main__":
    job_id = os.getenv("JOB_ID")
    dot_radius = int(os.getenv("DOT_RADIUS", "5"))

    if not job_id:
        print("No JOB_ID")
        exit(0)

    process_job(job_id, dot_radius)
