# worker.py
# Background worker for AI People Reader
# - Polls S3 "jobs/pending/" for JSON jobs
# - Moves job JSON -> "jobs/processing/"
# - Downloads input video from S3
# - Creates Johansson-dot or skeleton video with MediaPipe
# - Uploads result to "jobs/output/<job_id>/..."
# - Deletes processing JSON on success, moves to "jobs/failed/" on error

import os
import time
import json
import logging
import tempfile
import shutil

import boto3
from botocore.exceptions import ClientError, BotoCoreError

import cv2
import numpy as np
import mediapipe as mp

from config import (
    AWS_BUCKET,
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    OUTPUT_PREFIX,
    FAILED_PREFIX,
    REGION_NAME,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai-people-reader-worker")

# -----------------------------------------------------------------------------
# S3 client
# -----------------------------------------------------------------------------

s3 = boto3.client("s3", region_name=REGION_NAME)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def list_pending_jobs(max_keys: int = 1):
    """
    Return a list of pending job keys (JSON files) under PENDING_PREFIX.
    Only return at most `max_keys` keys.
    """
    try:
        resp = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=PENDING_PREFIX,
            MaxKeys=max_keys,
        )
    except (ClientError, BotoCoreError) as e:
        logger.error("Error listing pending jobs: %s", e, exc_info=True)
        return []

    contents = resp.get("Contents", [])
    # Filter out "folders" (keys ending with "/")
    keys = [obj["Key"] for obj in contents if not obj["Key"].endswith("/")]
    return keys


def move_s3_object(src_key: str, dst_key: str):
    """
    Move an object in S3: copy then delete.
    """
    logger.info("Claim job: %s -> %s", src_key, dst_key)
    try:
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": src_key},
            Key=dst_key,
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)
    except (ClientError, BotoCoreError) as e:
        logger.error("Error moving object %s -> %s: %s", src_key, dst_key, e)
        raise


def download_s3_file(key: str, local_path: str):
    """
    Download a file from S3 to local path.
    """
    uri = f"s3://{AWS_BUCKET}/{key}"
    logger.info("Downloading %s -> %s", uri, local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(AWS_BUCKET, key, local_path)


def upload_s3_file(local_path: str, key: str, content_type: str = "video/mp4"):
    """
    Upload a local file to S3.
    """
    uri = f"s3://{AWS_BUCKET}/{key}"
    logger.info("Uploading %s -> %s", local_path, uri)
    extra = {"ContentType": content_type} if content_type else {}
    s3.upload_file(local_path, AWS_BUCKET, key, ExtraArgs=extra)


def job_id_from_key(key: str) -> str:
    """
    Extract job_id from a JSON key like:
    jobs/pending/20260112_151553__109d9b.json
    """
    base = os.path.basename(key)
    job_id, _ = os.path.splitext(base)
    return job_id


# -----------------------------------------------------------------------------
# Video processing (Johansson dots + optional skeleton)
# -----------------------------------------------------------------------------

mp_pose = mp.solutions.pose
POSE_LANDMARKS = mp_pose.PoseLandmark


def _create_writer(out_path: str, width: int, height: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(out_path, fourcc, fps, (width, height))


def create_johansson_video(input_path: str, output_path: str):
    """
    Create a Johansson-dot style video:
    - Black background
    - White dots at pose landmarks detected by MediaPipe
    """
    logger.info("Creating Johansson dots video: %s -> %s", input_path, output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    out = _create_writer(output_path, width, height, fps)

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
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # Start from black frame
            dots_frame = np.zeros_like(frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in results.pose_landmarks.landmark:
                    if lm.visibility < 0.4:
                        continue
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(dots_frame, (cx, cy), 6, (255, 255, 255), -1)

            out.write(dots_frame)

            if frame_idx % 100 == 0:
                logger.info("Processed %d frames...", frame_idx)

    cap.release()
    out.release()
    logger.info("Johansson dots video created: %s", output_path)


def create_skeleton_video(input_path: str, output_path: str):
    """
    Optional: skeleton overlay video.
    """
    logger.info("Creating skeleton video: %s -> %s", input_path, output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    out = _create_writer(output_path, width, height, fps)

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

            if results.pose_landmarks:
                h, w, _ = frame.shape
                pts = []
                for lm in results.pose_landmarks.landmark:
                    if lm.visibility < 0.4:
                        pts.append(None)
                        continue
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    pts.append((cx, cy))

                # Draw simple skeleton: head–shoulders–hips–knees–ankles
                def draw_pair(a, b):
                    if a is None or b is None:
                        return
                    cv2.line(frame, a, b, (255, 255, 255), 3)

                # Indices from MediaPipe's PoseLandmark enum
                L = POSE_LANDMARKS
                idx = lambda lm: lm.value

                # Torso
                draw_pair(pts[idx(L.LEFT_SHOULDER)], pts[idx(L.RIGHT_SHOULDER)])
                draw_pair(pts[idx(L.LEFT_HIP)], pts[idx(L.RIGHT_HIP)])
                draw_pair(pts[idx(L.LEFT_SHOULDER)], pts[idx(L.LEFT_HIP)])
                draw_pair(pts[idx(L.RIGHT_SHOULDER)], pts[idx(L.RIGHT_HIP)])

                # Left leg
                draw_pair(pts[idx(L.LEFT_HIP)], pts[idx(L.LEFT_KNEE)])
                draw_pair(pts[idx(L.LEFT_KNEE)], pts[idx(L.LEFT_ANKLE)])

                # Right leg
                draw_pair(pts[idx(L.RIGHT_HIP)], pts[idx(L.RIGHT_KNEE)])
                draw_pair(pts[idx(L.RIGHT_KNEE)], pts[idx(L.RIGHT_ANKLE)])

                # Left arm
                draw_pair(pts[idx(L.LEFT_SHOULDER)], pts[idx(L.LEFT_ELBOW)])
                draw_pair(pts[idx(L.LEFT_ELBOW)], pts[idx(L.LEFT_WRIST)])

                # Right arm
                draw_pair(pts[idx(L.RIGHT_SHOULDER)], pts[idx(L.RIGHT_ELBOW)])
                draw_pair(pts[idx(L.RIGHT_ELBOW)], pts[idx(L.RIGHT_WRIST)])

            out.write(frame)

            if frame_idx % 100 == 0:
                logger.info("Processed %d frames (skeleton)...", frame_idx)

    cap.release()
    out.release()
    logger.info("Skeleton video created: %s", output_path)


# -----------------------------------------------------------------------------
# Job processing
# -----------------------------------------------------------------------------

def process_single_job(pending_key: str):
    """
    Process a single job:
    1) Move JSON from pending -> processing
    2) Parse JSON (must include `video_key`, optional `mode`)
    3) Download input video
    4) Run Johansson or skeleton processing
    5) Upload output
    6) Delete processing JSON on success
    """
    job_id = job_id_from_key(pending_key)
    processing_key = f"{PROCESSING_PREFIX}{job_id}.json"
    failed_key = f"{FAILED_PREFIX}{job_id}.json"

    # Move job file into "processing"
    move_s3_object(pending_key, processing_key)

    # Create temp dir per job
    tmp_dir = tempfile.mkdtemp(prefix=f"{job_id}_")
    logger.info("Working directory: %s", tmp_dir)

    job_json_path = os.path.join(tmp_dir, "job.json")
    input_video_path = os.path.join(tmp_dir, "input.mp4")
    output_video_path = os.path.join(tmp_dir, "output.mp4")

    try:
        # Download job JSON
        download_s3_file(processing_key, job_json_path)

        with open(job_json_path, "r", encoding="utf-8") as f:
            job_data = json.load(f)

        video_key = job_data["video_key"]
        mode = job_data.get("mode", "dots")

        logger.info(
            "Start job %s: video_key=%s, mode=%s",
            job_id,
            video_key,
            mode,
        )

        # Download source video
        download_s3_file(video_key, input_video_path)

        # Process according to mode
        if mode == "skeleton":
            create_skeleton_video(input_video_path, output_video_path)
            output_name = "skeleton.mp4"
        else:
            # default = dots
            create_johansson_video(input_video_path, output_video_path)
            output_name = "dots.mp4"

        # Upload result
        output_key = f"{OUTPUT_PREFIX}{job_id}/{output_name}"
        upload_s3_file(output_video_path, output_key, content_type="video/mp4")

        # Delete processing JSON now that we are done
        logger.info("Deleting processing job file: %s", processing_key)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)

        logger.info("Job %s finished successfully ✅", job_id)

    except Exception as e:
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)

        # Move JSON from processing -> failed for later inspection
        try:
            move_s3_object(processing_key, failed_key)
        except Exception:
            logger.error(
                "Failed to move processing job %s to failed prefix", job_id,
                exc_info=True,
            )
        raise
    finally:
        # Clean temp directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            logger.warning("Could not clean temp dir %s", tmp_dir, exc_info=True)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    logger.info("AI People Reader worker starting...")
    logger.info("Bucket=%s, region=%s", AWS_BUCKET, REGION_NAME)

    while True:
        try:
            keys = list_pending_jobs(max_keys=1)
            if not keys:
                logger.info("No pending jobs. Sleeping 5s...")
                time.sleep(5)
                continue

            # Process first pending job
            pending_key = keys[0]
            logger.info("Found pending job: %s", pending_key)
            process_single_job(pending_key)

        except KeyboardInterrupt:
            logger.info("Worker interrupted, shutting down.")
            break
        except Exception as e:
            logger.error("Unexpected error in main loop: %s", e, exc_info=True)
            # Small sleep to avoid hot loop on repeated errors
            time.sleep(5)


if __name__ == "__main__":
    main()
