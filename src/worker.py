# src/worker.py

import os
import time
import json
import logging
import tempfile
from typing import Optional, Dict

import boto3
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip

from config import (
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    OUTPUT_PREFIX,
    FAILED_PREFIX,
)

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# S3 setup
# -------------------------------------------------------------------
AWS_BUCKET = os.environ["AWS_BUCKET"]
REGION_NAME = os.getenv("REGION_NAME", "ap-southeast-1")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

s3 = boto3.client("s3", region_name=REGION_NAME)

# -------------------------------------------------------------------
# Helper: S3 Job Management
# -------------------------------------------------------------------
def find_next_pending_job() -> Optional[str]:
    """
    Return the key of the first pending job json, or None if none.
    """
    resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    contents = resp.get("Contents", [])
    for obj in contents:
        key = obj["Key"]
        if key.endswith(".json"):
            return key
    return None


def move_key(key: str, new_prefix: str) -> str:
    """
    Move a key to new_prefix within the same bucket.
    Returns the new key.
    """
    filename = key.split("/")[-1]
    new_key = f"{new_prefix}{filename}"

    logger.info(f"Moving {key} -> {new_key}")
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": key},
        Key=new_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=key)
    return new_key


def download_json(key: str) -> Dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        s3.download_file(AWS_BUCKET, key, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.remove(tmp_path)
    return data


def upload_json(key: str, payload: Dict) -> None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp.flush()
        tmp_path = tmp.name

    s3.upload_file(tmp_path, AWS_BUCKET, key)
    os.remove(tmp_path)


# -------------------------------------------------------------------
# Johansson dots generation
# -------------------------------------------------------------------
mp_pose = mp.solutions.pose


def create_johansson_dots_video(input_path: str, output_path: str) -> None:
    """
    Read input video, run Mediapipe Pose, and write a Johansson dots video
    (black background + white dots at each landmark).
    """
    logger.info(f"Creating Johansson dots video: {input_path} -> {output_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Black canvas
            canvas = np.zeros_like(frame)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

            out.write(canvas)

    cap.release()
    out.release()
    logger.info("Johansson dots video created successfully.")


# -------------------------------------------------------------------
# Job processing
# -------------------------------------------------------------------
def process_job(job_key: str) -> None:
    """
    Process a single job:
    1. Move JSON from pending -> processing
    2. Download input video
    3. Generate dots video
    4. Upload to output prefix
    5. Optionally update job JSON
    """
    logger.info(f"Claiming job: {job_key}")
    processing_key = move_key(job_key, PROCESSING_PREFIX)

    job = download_json(processing_key)
    job_id = os.path.splitext(os.path.basename(job_key))[0]

    video_key = job["video_key"]  # e.g. "jobs/20260112_235130__abcd/input/input.mp4"
    mode = job.get("mode", "dots")

    logger.info(f"Job {job_id}: video_key={video_key}, mode={mode}")

    # Download input video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        s3.download_file(AWS_BUCKET, video_key, tmp_in.name)
        input_path = tmp_in.name

    # Temporary output
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name

    try:
        # Right now we only support "dots" mode
        create_johansson_dots_video(input_path, output_path)

        # Upload to output directory
        output_dir = f"{OUTPUT_PREFIX}{job_id}/"
        output_key = f"{output_dir}dots.mp4"

        logger.info(f"Uploading result to s3://{AWS_BUCKET}/{output_key}")
        s3.upload_file(output_path, AWS_BUCKET, output_key)

        # Optionally update job json with output info
        job["status"] = "done"
        job["output_key"] = output_key
        upload_json(processing_key, job)

        logger.info(f"Job {job_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        # Move job JSON to failed prefix
        move_key(processing_key, FAILED_PREFIX)
        raise

    finally:
        # Cleanup temp files
        try:
            os.remove(input_path)
        except Exception:
            pass
        try:
            os.remove(output_path)
        except Exception:
            pass


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------
def main() -> None:
    logger.info("Worker starting. Bucket=%s, region=%s", AWS_BUCKET, REGION_NAME)

    while True:
        try:
            job_key = find_next_pending_job()
            if not job_key:
                logger.info("No pending jobs. Sleeping %s seconds…", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
                continue

            process_job(job_key)

        except Exception:
            # Already logged inside process_job; just wait a bit
            logger.exception("Unhandled error in main loop.")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
