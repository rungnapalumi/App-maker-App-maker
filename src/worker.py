# src/worker.py

import os
import io
import json
import time
import logging
import tempfile
from typing import Optional

import boto3
import botocore
import cv2
import numpy as np
import mediapipe as mp

from config import (
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    OUTPUT_PREFIX,
    FAILED_PREFIX,
)

# --------------------------------------------------
# Logging setup
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# AWS / S3 setup
# --------------------------------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET", os.getenv("BUCKET", "ai-people-reader-storage"))
AWS_REGION = os.getenv("AWS_REGION", os.getenv("REGION_NAME", "ap-southeast-1"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))  # seconds

s3 = boto3.client("s3", region_name=AWS_REGION)

logger.info(
    "Worker starting. Bucket=%s, region=%s, poll_interval=%ss",
    AWS_BUCKET,
    AWS_REGION,
    POLL_INTERVAL,
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------


def list_pending_job_keys() -> list[str]:
    """Return list of JSON job keys under the pending prefix."""
    try:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    except botocore.exceptions.BotoCoreError as e:
        logger.error("Error listing pending jobs: %s", e)
        return []

    contents = resp.get("Contents", [])
    keys = [obj["Key"] for obj in contents if obj["Key"].endswith(".json")]
    return keys


def move_object(src_key: str, dest_key: str) -> None:
    """Move an object within the same bucket (copy + delete)."""
    if src_key == dest_key:
        return

    copy_source = {"Bucket": AWS_BUCKET, "Key": src_key}
    s3.copy_object(Bucket=AWS_BUCKET, CopySource=copy_source, Key=dest_key)
    s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)


def claim_next_job() -> Optional[str]:
    """
    Claim one job:
    - pick first JSON in pending
    - move to processing
    - return processing key
    """
    pending_keys = list_pending_job_keys()
    if not pending_keys:
        return None

    pending_key = sorted(pending_keys)[0]  # deterministic
    job_id = os.path.basename(pending_key)

    processing_key = os.path.join(PROCESSING_PREFIX, job_id)
    logger.info("Claim job %s: %s -> %s", job_id, pending_key, processing_key)

    move_object(pending_key, processing_key)
    return processing_key


def download_s3_to_temp(key: str, suffix: str) -> str:
    """Download S3 object to a temporary file and return the local path."""
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # we will reopen via OpenCV
    logger.info("Downloading s3://%s/%s -> %s", AWS_BUCKET, key, tmp_path)
    s3.download_file(AWS_BUCKET, key, tmp_path)
    return tmp_path


def upload_file_to_s3(local_path: str, key: str, content_type: str = "video/mp4") -> None:
    """Upload local file to S3."""
    logger.info("Uploading %s -> s3://%s/%s", local_path, AWS_BUCKET, key)
    extra = {"ContentType": content_type}
    s3.upload_file(local_path, AWS_BUCKET, key, ExtraArgs=extra)


# --------------------------------------------------
# Johansson dots generator (using MediaPipe Pose)
# --------------------------------------------------


def create_johansson_dots_video(input_path: str, output_path: str) -> None:
    """
    Read input video, run MediaPipe Pose, and render white dots on black background.
    Save as MP4 at output_path.
    """
    logger.info("Creating Johansson dots: %s -> %s", input_path, output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # black background
            dots_frame = np.zeros_like(frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in results.pose_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(dots_frame, (x, y), 4, (255, 255, 255), -1)

            out.write(dots_frame)

    cap.release()
    out.release()
    logger.info("Johansson dots video created successfully.")


# --------------------------------------------------
# Job processing
# --------------------------------------------------


def process_job(processing_key: str) -> None:
    """
    processing_key: jobs/processing/<job_id>.json

    Expected job JSON:
    {
        "video_key": "jobs/<job_id>/input/input.mp4",
        "mode": "dots"   # future: could support 'skeleton' etc.
    }
    """
    job_id = os.path.splitext(os.path.basename(processing_key))[0]
    logger.info("Processing job %s (%s)", job_id, processing_key)

    # Download and parse job JSON
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=processing_key)
    job_data = json.loads(obj["Body"].read().decode("utf-8"))

    video_key = job_data.get("video_key")
    mode = job_data.get("mode", "dots")

    if not video_key:
        raise ValueError("Job JSON missing 'video_key'")

    # Download video
    input_path = download_s3_to_temp(video_key, suffix=".mp4")

    # Prepare local output
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    if mode == "dots":
        create_johansson_dots_video(input_path, output_path)
        output_key = os.path.join(OUTPUT_PREFIX, job_id, "dots.mp4")
    else:
        # สำหรับอนาคต ถ้าจะมีโหมดอื่น
        raise ValueError(f"Unsupported mode: {mode}")

    # Upload result
    upload_file_to_s3(output_path, output_key)

    logger.info("Job %s done. Output=%s", job_id, output_key)


def mark_job_failed(processing_key: str, error_message: str) -> None:
    """Move job JSON to failed/ และเก็บ error message เพิ่มเติม."""
    job_id = os.path.basename(processing_key)
    failed_key = os.path.join(FAILED_PREFIX, job_id)

    try:
        # Read current JSON
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=processing_key)
        job_data = json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        job_data = {}

    job_data["status"] = "failed"
    job_data["error"] = error_message

    body = json.dumps(job_data).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=failed_key, Body=body, ContentType="application/json")
    s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)

    logger.error("Marked job failed: %s (%s)", job_id, error_message)


# --------------------------------------------------
# Main loop
# --------------------------------------------------


def main() -> None:
    while True:
        try:
            processing_key = claim_next_job()
            if not processing_key:
                logger.info("No pending jobs. Sleeping %ss...", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
                continue

            try:
                process_job(processing_key)
                # ถ้า success สามารถลบ job JSON ใน processing ได้ หรือจะเก็บไว้ก็ได้
                s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)
            except Exception as e:
                logger.exception("Error while processing job %s", processing_key)
                mark_job_failed(processing_key, str(e))

        except Exception:
            # อย่าให้ worker ตายง่าย ๆ
            logger.exception("Unexpected error in main loop")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
