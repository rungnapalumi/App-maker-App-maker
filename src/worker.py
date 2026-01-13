"""
worker.py — AI People Reader background worker

หน้าที่:
- ดึง "งาน" (.json) จากโฟลเดอร์ S3: jobs/pending/
- ย้ายไป jobs/processing/ (เพื่อถือว่ากำลังทำ)
- ดาวน์โหลด input video ตามที่ระบุใน json
- แปลงเป็น Johansson dots video ด้วย Mediapipe Pose
- อัปโหลดผลลัพธ์ไป jobs/output/{job_id}/dots.mp4
- ถ้าพลาด -> ย้าย job json ไป jobs/failed/
"""

import os
import io
import json
import time
import logging
import tempfile
from typing import Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

from config import (
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    OUTPUT_PREFIX,
    FAILED_PREFIX,
)

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Environment / S3 client
# ---------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-southeast-1"))

if not AWS_BUCKET:
    logger.error("Environment variable AWS_BUCKET is not set. Worker cannot start.")
    raise SystemExit(1)

logger.info(f"Using bucket: {AWS_BUCKET}, region: {AWS_REGION}")

s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------
# Helper functions for S3 and job management
# ---------------------------------------------------------

def list_pending_jobs(max_jobs: int = 5) -> list[str]:
    """
    List pending job JSON keys under PENDING_PREFIX.
    Returns a list of S3 keys.
    """
    keys: list[str] = []
    try:
        resp = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=PENDING_PREFIX,
            MaxKeys=max_jobs,
        )
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Error listing pending jobs: {e}")
        return keys

    contents = resp.get("Contents", [])
    for obj in contents:
        key = obj["Key"]
        if key.endswith(".json"):
            keys.append(key)

    return keys


def move_object(src_key: str, dst_key: str) -> None:
    """
    Move an object within the same bucket by copy + delete.
    """
    try:
        logger.info(f"Moving {src_key} -> {dst_key}")
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": src_key},
            Key=dst_key,
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Error moving {src_key} to {dst_key}: {e}")
        raise


def claim_job(pending_key: str) -> str:
    """
    Claim a job by moving it from pending/ to processing/.
    Returns the processing key.
    """
    filename = os.path.basename(pending_key)
    processing_key = os.path.join(PROCESSING_PREFIX, filename)

    move_object(pending_key, processing_key)
    logger.info(f"Claimed job: {pending_key} -> {processing_key}")
    return processing_key


def load_job_json(key: str) -> dict:
    """
    Download and parse the job JSON file from S3.
    """
    try:
        resp = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        data = resp["Body"].read()
        return json.loads(data.decode("utf-8"))
    except (BotoCoreError, ClientError, json.JSONDecodeError) as e:
        logger.error(f"Error loading job JSON {key}: {e}")
        raise


def upload_file(local_path: str, s3_key: str) -> None:
    """
    Upload a local file to S3 at s3_key.
    """
    try:
        logger.info(f"Uploading {local_path} -> s3://{AWS_BUCKET}/{s3_key}")
        s3.upload_file(local_path, AWS_BUCKET, s3_key)
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Error uploading {local_path} to {s3_key}: {e}")
        raise


def mark_failed(processing_key: str, reason: str) -> None:
    """
    Move processing job JSON to failed/ and log the reason.
    """
    filename = os.path.basename(processing_key)
    failed_key = os.path.join(FAILED_PREFIX, filename)
    logger.error(f"Marking job as FAILED: {processing_key} -> {failed_key} ({reason})")
    try:
        move_object(processing_key, failed_key)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error moving job to failed/: {e}")


# ---------------------------------------------------------
# Johansson dots generation
# ---------------------------------------------------------

def load_video_metadata(path: str) -> Tuple[int, int, float, int]:
    """
    Return (width, height, fps, frame_count) for a given video file.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cap.release()
    return width, height, fps, frame_count


def create_johansson_dots_video(
    input_path: str,
    output_path: str,
    dot_radius: int = 6,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> None:
    """
    Read input video, run Mediapipe Pose per frame,
    and create a black-background video with white dots
    at all pose landmarks (Johansson-style).

    Requires mediapipe + opencv to be installed.
    """
    if not MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed in this environment.")

    width, height, fps, _ = load_video_metadata(input_path)
    logger.info(
        f"Creating Johansson dots: {input_path} -> {output_path} | "
        f"{width}x{height} @ {fps:.2f} fps"
    )

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

    mp_pose = mp.solutions.pose

    frame_idx = 0
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # Black background
            dot_frame = np.zeros_like(frame)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    x_px = int(lm.x * width)
                    y_px = int(lm.y * height)
                    if 0 <= x_px < width and 0 <= y_px < height:
                        cv2.circle(dot_frame, (x_px, y_px), dot_radius, (255, 255, 255), -1)

            out.write(dot_frame)

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    logger.info("Johansson dots video created successfully.")


# ---------------------------------------------------------
# Job processing logic
# ---------------------------------------------------------

def download_s3_to_tempfile(key: str, suffix: str = "") -> str:
    """
    Download an S3 object to a temporary file and return its path.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        logger.info(f"Downloading s3://{AWS_BUCKET}/{key} -> {path}")
        s3.download_file(AWS_BUCKET, key, path)
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Error downloading {key}: {e}")
        os.remove(path)
        raise

    return path


def process_job(processing_key: str) -> None:
    """
    Handle a single job in processing/.
    """
    job_id = os.path.splitext(os.path.basename(processing_key))[0]
    logger.info(f"Processing job: {job_id}")

    try:
        job = load_job_json(processing_key)
    except Exception as e:  # noqa: BLE001
        mark_failed(processing_key, f"Invalid job json: {e}")
        return

    video_key = job.get("video_key")
    mode = job.get("mode", "dots")

    if not video_key:
        mark_failed(processing_key, "Missing 'video_key' in job json.")
        return

    logger.info(f"Job {job_id}: video_key={video_key}, mode={mode}")

    # Download input video
    try:
        input_path = download_s3_to_tempfile(video_key, suffix=".mp4")
    except Exception as e:  # noqa: BLE001
        mark_failed(processing_key, f"Failed to download input video: {e}")
        return

    # Prepare temp output path
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        if mode == "dots":
            create_johansson_dots_video(input_path, output_path)
        else:
            logger.warning(
                f"Unknown mode '{mode}' for job {job_id}. "
                f"Defaulting to Johansson dots."
            )
            create_johansson_dots_video(input_path, output_path)

        # Upload result
        output_key = os.path.join(OUTPUT_PREFIX, job_id, "dots.mp4")
        upload_file(output_path, output_key)

        logger.info(f"Job {job_id} completed. Output: s3://{AWS_BUCKET}/{output_key}")

        # Optionally, we could delete the processing json or keep it as history.
        # For now, move it to output/ as well for record.
        finished_json_key = os.path.join(OUTPUT_PREFIX, job_id, f"{job_id}.json")
        move_object(processing_key, finished_json_key)

    except Exception as e:  # noqa: BLE001
        logger.exception(f"Error while processing job {job_id}: {e}")
        mark_failed(processing_key, str(e))
    finally:
        # Clean up temp files
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except Exception:
            pass
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception:
            pass


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

POLL_INTERVAL_SECONDS = 5


def main() -> None:
    """
    Main worker loop:
    - list pending jobs
    - claim & process one by one
    - sleep if nothing to do
    """
    if not MP_AVAILABLE:
        logger.error(
            "mediapipe is NOT available. "
            "Please ensure 'mediapipe' (or mediapipe-lite with correct imports) "
            "is installed in this environment."
        )

    if not MOVIEPY_AVAILABLE:
        logger.warning(
            "moviepy is not available. This worker will still run using OpenCV only, "
            "but make sure requirements.txt includes 'moviepy' if needed elsewhere."
        )

    logger.info("AI People Reader worker started. Waiting for jobs...")

    while True:
        try:
            pending = list_pending_jobs()
            if not pending:
                logger.info("No pending jobs. Sleeping...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info(f"Found {len(pending)} pending job(s).")
            for pending_key in pending:
                try:
                    processing_key = claim_job(pending_key)
                    process_job(processing_key)
                except Exception as e:  # noqa: BLE001
                    # If claiming itself fails badly, just log and continue.
                    logger.exception(f"Unexpected error handling job {pending_key}: {e}")

        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down.")
            break
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Unexpected top-level error: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
