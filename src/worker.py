# src/worker.py
# ============================================
# AI People Reader — Worker (Johansson DOTS ONLY)
# - MediaPipe Pose
# - DOT radius = 5 px
# - NO skeleton lines
# - NO audio
# - Safe for Render / Linux
# ============================================

import os
import json
import time
import uuid
import tempfile
import cv2
import boto3
import numpy as np
from datetime import datetime, timezone

# ✅ CORRECT MediaPipe import (THIS IS THE FIX)
import mediapipe as mp

# -----------------------
# Config
# -----------------------
DOT_RADIUS = 5
POSE_CONFIDENCE = 0.5

AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("S3_BUCKET")

assert AWS_REGION, "AWS_REGION missing"
assert S3_BUCKET, "S3_BUCKET missing"

s3 = boto3.client("s3", region_name=AWS_REGION)

mp_pose = mp.solutions.pose


# -----------------------
# Utils
# -----------------------
def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def strip_audio(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


# -----------------------
# Core DOT renderer
# -----------------------
def render_johansson_dots(input_video, output_video):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=POSE_CONFIDENCE,
        min_tracking_confidence=POSE_CONFIDENCE,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            blank = np.zeros_like(frame)

            if result.pose_landmarks:
                for lm in result.pose_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            blank,
                            (cx, cy),
                            DOT_RADIUS,
                            (255, 255, 255),
                            -1,
                        )

            out.write(blank)

    cap.release()
    out.release()


# -----------------------
# Job handler
# -----------------------
def handle_job(job):
    job_id = job["job_id"]
    input_key = job["input_key"]

    log(f"Processing job {job_id}")

    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "input_raw.mp4")
        no_audio = os.path.join(tmp, "input_noaudio.mp4")
        output = os.path.join(tmp, "overlay.mp4")

        # download
        s3.download_file(S3_BUCKET, input_key, raw)

        # strip audio
        strip_audio(raw, no_audio)

        # render dots
        render_johansson_dots(no_audio, output)

        out_key = f"jobs/{job_id}/output/overlay.mp4"
        s3.upload_file(output, S3_BUCKET, out_key)

    return {
        "overlay_key": out_key
    }


# -----------------------
# Worker loop
# -----------------------
def main():
    log("Worker boot (Johansson DOTS ONLY)")
    log(f"AWS_REGION={AWS_REGION}")
    log(f"S3_BUCKET={S3_BUCKET}")
    s3.head_bucket(Bucket=S3_BUCKET)
    log("S3 reachable ✅")

    while True:
        # ⬇️ replace this with your actual job queue pull
        time.sleep(5)
        log("heartbeat (no pending jobs)")


if __name__ == "__main__":
    main()
