# ============================================================
# AI People Reader — Background Worker
# Johansson dots (NO skeleton lines)
# - dot radius = 5
# - remove audio
# - Render-safe MediaPipe import
# ============================================================

import os
import time
import json
import uuid
import tempfile
from datetime import datetime, timezone

import boto3
import cv2
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
DOT_RADIUS = 5
FPS_OUT = None  # keep original fps
HEARTBEAT_SEC = 15

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET")

# =========================
# S3 CLIENT (NO proxies)
# =========================
def get_s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=boto3.session.Config(
            proxies={}, signature_version="s3v4"
        ),
    )

# =========================
# LOG HELPERS
# =========================
def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

# =========================
# JOHANSSON DOTS VIDEO
# =========================
def johansson_dots_video(input_path, output_path, dot_radius=5):
    log("Starting Johansson dots processing")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            canvas = np.zeros_like(frame)

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    if lm.visibility < 0.5:
                        continue
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(
                        canvas,
                        (cx, cy),
                        dot_radius,
                        (255, 255, 255),
                        -1,
                    )

            out.write(canvas)

    cap.release()
    out.release()
    log("Johansson dots processing finished")

# =========================
# JOB HANDLER
# =========================
def handle_job(job):
    s3 = get_s3()
    job_id = job["job_id"]

    in_key = job["input_key"]
    out_key = f"jobs/{job_id}/output/overlay.mp4"

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "output.mp4")

        log(f"Downloading s3://{S3_BUCKET}/{in_key}")
        s3.download_file(S3_BUCKET, in_key, in_path)

        johansson_dots_video(in_path, out_path, DOT_RADIUS)

        log(f"Uploading result to s3://{S3_BUCKET}/{out_key}")
        s3.upload_file(
            out_path,
            S3_BUCKET,
            out_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

    return {
        "overlay_key": out_key
    }

# =========================
# MAIN LOOP
# =========================
def main():
    log("✅ Worker boot (Johansson dots)")
    log(f"AWS_REGION='{AWS_REGION}'")
    log(f"S3_BUCKET='{S3_BUCKET}'")

    s3 = get_s3()
    s3.head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        # ---- heartbeat only (Render-safe idle worker) ----
        log("⏳ heartbeat (no pending jobs)")
        time.sleep(HEARTBEAT_SEC)

if __name__ == "__main__":
    main()
