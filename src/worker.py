# src/worker.py
"""
Background worker for AI People Reader — Johansson dots renderer.

Loop:
1. Poll S3 for pending job JSON files under PENDING_PREFIX.
2. Claim a job by copying JSON:
      jobs/pending/<job_id>.json  ->  jobs/processing/<job_id>.json
3. Download the input video from the S3 key in the job JSON.
4. Render Johansson-style dots video with MediaPipe Pose.
5. Upload overlay video to S3 and update job JSON with status/output.
6. Move JSON to jobs/output/<job_id>.json or jobs/failed/<job_id>.json.
"""

import os
import io
import json
import time
import tempfile
import logging
from typing import Dict, Any, Optional, List

import boto3
from botocore.exceptions import ClientError

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

from config import (
    AWS_BUCKET,
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    FAILED_PREFIX,
    OUTPUT_PREFIX,
    REGION_NAME,
)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

logger = logging.getLogger("worker")


# -----------------------------------------------------------------------------
# S3 helpers
# -----------------------------------------------------------------------------

s3 = boto3.client("s3", region_name=REGION_NAME)


def list_pending_jobs() -> List[str]:
    """Return list of S3 keys for pending job JSONs."""
    try:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    except ClientError as e:
        logger.error("Error listing pending jobs: %s", e)
        return []

    contents = resp.get("Contents", [])
    keys = [obj["Key"] for obj in contents if obj["Key"].endswith(".json")]
    return keys


def claim_job(pending_key: str) -> Optional[str]:
    """
    Move job JSON from pending -> processing using CopyObject + DeleteObject.

    Returns the processing key if successful, otherwise None.
    """
    job_file_name = pending_key.split("/")[-1]
    processing_key = f"{PROCESSING_PREFIX}{job_file_name}"

    logger.info("Claim job %s: %s -> %s", job_file_name, pending_key, processing_key)

    copy_source = {"Bucket": AWS_BUCKET, "Key": pending_key}

    try:
        # Copy JSON to processing/
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource=copy_source,  # ✅ Correct format: dict with Bucket + Key
            Key=processing_key,
        )
        # Remove from pending/
        s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)
    except ClientError as e:
        logger.error("Failed to claim job %s: %s", pending_key, e)
        return None

    return processing_key


def download_s3_file(key: str, local_path: str) -> None:
    logger.info("Downloading s3://%s/%s -> %s", AWS_BUCKET, key, local_path)
    s3.download_file(AWS_BUCKET, key, local_path)


def upload_s3_file(local_path: str, key: str, content_type: str = "video/mp4") -> None:
    logger.info("Uploading %s -> s3://%s/%s", local_path, AWS_BUCKET, key)
    s3.upload_file(
        local_path,
        AWS_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def load_job_json(key: str) -> Dict[str, Any]:
    logger.info("Loading job JSON from s3://%s/%s", AWS_BUCKET, key)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


def save_job_json(key: str, data: Dict[str, Any]) -> None:
    logger.info("Saving job JSON to s3://%s/%s", AWS_BUCKET, key)
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )


def move_job_json(current_key: str, target_prefix: str) -> str:
    """
    Move job JSON from current_key to target_prefix/<file_name>.json.
    Returns the new key.
    """
    job_file_name = current_key.split("/")[-1]
    new_key = f"{target_prefix}{job_file_name}"

    logger.info("Move job JSON: %s -> %s", current_key, new_key)

    copy_source = {"Bucket": AWS_BUCKET, "Key": current_key}

    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource=copy_source,
        Key=new_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=current_key)

    return new_key


# -----------------------------------------------------------------------------
# Johansson dots rendering
# -----------------------------------------------------------------------------

JOHANSSON_LANDMARKS = [
    0,   # nose
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
]


def render_johansson_dots(
    input_video: str,
    output_video: str,
    dot_radius: int = 5,
    max_dots: int = 30,
) -> None:
    """
    Create Johansson-style white-dot-on-black video from input using MediaPipe Pose.
    """
    if mp is None:
        raise RuntimeError(
            "mediapipe is not available in this environment. "
            "Check requirements.txt and Python version."
        )

    logger.info(
        "Rendering Johansson dots: input=%s, output=%s, radius=%d, max_dots=%d",
        input_video,
        output_video,
        dot_radius,
        max_dots,
    )

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

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

            h, w, _ = frame.shape
            black = np.zeros_like(frame)

            # Run pose
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark

                points = []
                for idx in JOHANSSON_LANDMARKS:
                    if idx < len(lm):
                        p = lm[idx]
                        if p.visibility > 0.5:
                            x = int(p.x * w)
                            y = int(p.y * h)
                            points.append((x, y))

                if max_dots > 0:
                    points = points[:max_dots]

                for (x, y) in points:
                    cv2.circle(
                        black,
                        (x, y),
                        dot_radius,
                        (255, 255, 255),
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

            out.write(black)

    cap.release()
    out.release()

    logger.info("Finished rendering Johansson dots -> %s", output_video)


# -----------------------------------------------------------------------------
# Job processing
# -----------------------------------------------------------------------------

def get_job_id_from_key(key: str) -> str:
    """Extract job_id from key like 'jobs/pending/<job_id>.json'."""
    filename = key.split("/")[-1]
    if filename.endswith(".json"):
        return filename[:-5]
    return filename


def process_single_job(processing_key: str) -> None:
    """
    Process one job JSON which is already in jobs/processing/.
    """
    job = load_job_json(processing_key)
    job_id = job.get("job_id") or get_job_id_from_key(processing_key)

    # Read parameters from JSON (be tolerant to naming)
    input_key = (
        job.get("input_video_key")
        or job.get("input_key")
        or job.get("video_key")
    )
    if not input_key:
        raise RuntimeError("Job JSON missing 'input_video_key' / 'input_key' / 'video_key'")

    dot_radius = int(job.get("dot_radius", 5))
    dot_count = int(job.get("dot_count", 30))

    # Decide output overlay key
    outputs = job.get("outputs") or {}
    overlay_key = (
        outputs.get("overlay_key")
        or job.get("overlay_key")
        or f"jobs/{job_id}/output/dots.mp4"
    )

    logger.info(
        "Processing job %s (input=%s, overlay_key=%s, radius=%d, count=%d)",
        job_id,
        input_key,
        overlay_key,
        dot_radius,
        dot_count,
    )

    # Update JSON to reflect processing status
    job["job_id"] = job_id
    job["status"] = "processing"
    job["progress"] = 10
    job["message"] = "Worker started processing."
    job.setdefault("outputs", {})["overlay_key"] = overlay_key
    save_job_json(processing_key, job)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_in = os.path.join(tmpdir, "input.mp4")
        local_out = os.path.join(tmpdir, "dots.mp4")

        # Download input
        download_s3_file(input_key, local_in)

        # Render dots
        job["progress"] = 60
        job["message"] = "Rendering Johansson dots overlay."
        save_job_json(processing_key, job)

        render_johansson_dots(
            input_video=local_in,
            output_video=local_out,
            dot_radius=dot_radius,
            max_dots=dot_count,
        )

        # Upload overlay video
        upload_s3_file(local_out, overlay_key, content_type="video/mp4")

    # Mark done
    job["status"] = "done"
    job["progress"] = 100
    job["message"] = "Completed"
    job["outputs"]["overlay_key"] = overlay_key
    save_job_json(processing_key, job)

    # Move JSON to output prefix
    move_job_json(processing_key, OUTPUT_PREFIX)

    logger.info("Job %s completed successfully.", job_id)


def fail_job(processing_key: str, error_message: str) -> None:
    """
    Update job JSON with failure info and move to FAILED_PREFIX.
    """
    job_id = get_job_id_from_key(processing_key)

    try:
        job = load_job_json(processing_key)
    except Exception:
        job = {"job_id": job_id}

    job["status"] = "failed"
    job["progress"] = 100
    job["message"] = error_message

    try:
        save_job_json(processing_key, job)
    except Exception as e:
        logger.error("Could not save failed job JSON: %s", e)

    try:
        move_job_json(processing_key, FAILED_PREFIX)
    except Exception as e:
        logger.error("Could not move failed job JSON: %s", e)

    logger.error("Job %s failed: %s", job_id, error_message)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 5


def main_loop() -> None:
    logger.info("AI People Reader worker started. Bucket=%s Region=%s", AWS_BUCKET, REGION_NAME)

    if mp is None:
        logger.error("mediapipe is NOT available. Jobs will fail with an error.")
        # เราไม่ exit ทันที เพื่อให้เห็น error ชัด ๆ ใน job JSON แต่แจ้งเตือนไว้ก่อน

    while True:
        try:
            pending_keys = list_pending_jobs()
        except Exception as e:
            logger.error("Error while listing pending jobs: %s", e)
            pending_keys = []

        if not pending_keys:
            logger.debug("No pending jobs. Sleeping %ds", POLL_INTERVAL_SECONDS)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        logger.info("Found %d pending job(s).", len(pending_keys))

        for pending_key in pending_keys:
            processing_key = claim_job(pending_key)
            if not processing_key:
                # claim ไม่สำเร็จ ข้ามไปตัวถัดไป
                continue

            try:
                process_single_job(processing_key)
            except Exception as e:
                logger.exception("Unhandled error while processing job %s", processing_key)
                fail_job(processing_key, str(e))

        # หลังจากลูป jobs รอบนี้เสร็จ พักสั้น ๆ ก่อนเช็คใหม่
        time.sleep(1)


if __name__ == "__main__":
    main_loop()
