# src/worker.py
# AI People Reader – background worker on Render
#
# - Polls S3 "pending" jobs
# - Moves them to "processing"
# - Downloads the input video from S3
# - Generates either:
#     * Johansson dots video (black background + white dots)
#     * Skeleton overlay video
# - Uploads result to the "output" prefix
# - On error, moves job JSON to "failed"

import os
import time
import json
import logging
import tempfile
from typing import Optional, Tuple, Dict

import boto3
from botocore.exceptions import ClientError

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

from config import (
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    OUTPUT_PREFIX,
    FAILED_PREFIX,
)

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# AWS / S3 config
# ---------------------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "5.0"))

if not AWS_BUCKET:
    raise RuntimeError("Environment variable AWS_BUCKET is required")

s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------------------------------------------------------------------
# S3 helper functions
# ---------------------------------------------------------------------


def list_next_pending_job_key() -> Optional[str]:
    """
    Returns the S3 key of the next pending job JSON, or None if there is none.
    """
    try:
        response = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=PENDING_PREFIX,
            MaxKeys=50,
        )
    except ClientError as e:
        logger.error("Error listing pending jobs: %s", e)
        return None

    contents = response.get("Contents", [])
    json_keys = sorted(
        [
            obj["Key"]
            for obj in contents
            if obj["Key"].endswith(".json")
        ]
    )
    if not json_keys:
        return None

    return json_keys[0]


def move_object(old_key: str, new_key: str) -> None:
    """
    Copy object within the same bucket and then delete the original.
    """
    try:
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": old_key},
            Key=new_key,
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)
    except ClientError as e:
        logger.error("Error moving %s -> %s: %s", old_key, new_key, e)
        raise


def claim_job(pending_key: str) -> Tuple[str, str, Dict]:
    """
    Move job JSON from pending to processing and return:
    - job_id
    - processing_key
    - job_spec (dict parsed from JSON)
    """
    job_filename = os.path.basename(pending_key)
    job_id, _ = os.path.splitext(job_filename)

    processing_key = pending_key.replace(PENDING_PREFIX, PROCESSING_PREFIX, 1)

    logger.info(
        "Claim job %s: %s -> %s",
        job_id,
        pending_key,
        processing_key,
    )
    move_object(pending_key, processing_key)

    # Load JSON
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=processing_key)
        body = obj["Body"].read()
        job_spec = json.loads(body.decode("utf-8"))
    except Exception as e:
        logger.error("Error reading job JSON %s: %s", processing_key, e)
        raise

    return job_id, processing_key, job_spec


def mark_job_failed(processing_key: str, reason: str) -> None:
    """
    Move job JSON from processing to failed and log the reason.
    """
    failed_key = processing_key.replace(PROCESSING_PREFIX, FAILED_PREFIX, 1)
    logger.error("Marking job failed (%s): %s -> %s", reason, processing_key, failed_key)
    try:
        move_object(processing_key, failed_key)
    except Exception:
        logger.exception("Failed to move job JSON to failed prefix")


def upload_output(job_id: str, local_path: str, filename: str) -> str:
    """
    Upload processed video to S3 and return the S3 key.
    """
    key = f"{OUTPUT_PREFIX}{job_id}/{filename}"
    logger.info("Uploading result to s3://%s/%s", AWS_BUCKET, key)
    s3.upload_file(local_path, AWS_BUCKET, key)
    return key


# ---------------------------------------------------------------------
# Video processing functions
# ---------------------------------------------------------------------


def create_johansson_dots(
    input_path: str,
    output_path: str,
    min_visibility: float = 0.4,
    dot_radius: int = 6,
) -> None:
    """
    Generate Johansson dots video: black background + white dots
    at Mediapipe pose landmark positions.
    """
    logger.info("Creating Johansson dots: %s -> %s", input_path, output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # start with pure black frame
            dots_frame = np.zeros_like(frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in results.pose_landmarks.landmark:
                    if lm.visibility is not None and lm.visibility < min_visibility:
                        continue
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(dots_frame, (x, y), dot_radius, (255, 255, 255), -1)

            out.write(dots_frame)
            frame_idx += 1

    cap.release()
    out.release()
    logger.info("Johansson dots created (%d frames)", frame_idx)


def create_skeleton_overlay(
    input_path: str,
    output_path: str,
) -> None:
    """
    Generate skeleton overlay video: original frame + pose skeleton lines.
    """
    logger.info("Creating skeleton overlay: %s -> %s", input_path, output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 255), thickness=2, circle_radius=2
                    ),
                )

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    logger.info("Skeleton overlay created (%d frames)", frame_idx)


# ---------------------------------------------------------------------
# Job processing
# ---------------------------------------------------------------------


def process_job(job_id: str, processing_key: str, job_spec: Dict) -> None:
    """
    Execute a single job given its ID, processing JSON key and spec.
    Expected JSON schema:

    {
      "video_key": "jobs/20260112_235130__abcd/input/input.mp4",
      "mode": "dots" | "skeleton"
    }
    """
    video_key = job_spec.get("video_key")
    mode = job_spec.get("mode")

    if not video_key or not mode:
        raise ValueError(f"Invalid job spec for {job_id}: {job_spec}")

    logger.info(
        "Processing job %s: video_key=%s, mode=%s", job_id, video_key, mode
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        local_input = os.path.join(tmpdir, "input.mp4")
        local_output = os.path.join(tmpdir, "output.mp4")

        # Download video from S3
        logger.info(
            "Downloading s3://%s/%s -> %s",
            AWS_BUCKET,
            video_key,
            local_input,
        )
        s3.download_file(AWS_BUCKET, video_key, local_input)

        # Run processing
        if mode == "dots":
            create_johansson_dots(local_input, local_output)
            output_filename = "dots.mp4"
        elif mode == "skeleton":
            create_skeleton_overlay(local_input, local_output)
            output_filename = "skeleton.mp4"
        else:
            raise ValueError(f"Unknown mode for job {job_id}: {mode}")

        # Upload result
        output_key = upload_output(job_id, local_output, output_filename)
        logger.info(
            "Job %s done. Result: s3://%s/%s",
            job_id,
            AWS_BUCKET,
            output_key,
        )

    # Job JSON stays in processing/; frontend can delete/clean up if needed
    # If you want to auto-clean successful jobs, you can delete processing_key here.
    # s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------


def main() -> None:
    logger.info(
        "Worker starting. Bucket=%s, region=%s, poll_interval=%.1fs",
        AWS_BUCKET,
        AWS_REGION,
        POLL_INTERVAL,
    )

    while True:
        try:
            pending_key = list_next_pending_job_key()
        except Exception:
            logger.exception("Unexpected error while listing pending jobs")
            pending_key = None

        if not pending_key:
            logger.info("No pending jobs. Sleeping %.1fs...", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)
            continue

        try:
            job_id, processing_key, job_spec = claim_job(pending_key)
        except Exception:
            logger.exception("Failed to claim job from %s", pending_key)
            time.sleep(POLL_INTERVAL)
            continue

        try:
            process_job(job_id, processing_key, job_spec)
        except Exception as e:
            logger.exception("Error processing job %s: %s", job_id, e)
            try:
                mark_job_failed(processing_key, str(e))
            except Exception:
                logger.exception(
                    "Additionally failed to mark job %s as failed", job_id
                )

        # loop and look for the next job


if __name__ == "__main__":
    main()
