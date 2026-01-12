import os
import json
import time
import tempfile
import logging
from typing import Dict, Any, List

import boto3
from botocore.exceptions import ClientError

import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# Logging setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger("worker")

# -----------------------------
# Environment
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET", "ai-people-reader-storage")
JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"

# MediaPipe Pose
mp_pose = mp.solutions.pose


# =============================
# S3 HELPERS
# =============================

def list_pending_jobs() -> List[Dict[str, Any]]:
    """Return list of pending job objects from S3."""
    try:
        resp = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=PENDING_PREFIX + "/",
        )
    except ClientError as e:
        logger.error(f"list_pending_jobs error: {e}")
        return []

    contents = resp.get("Contents", [])
    # filter only json files
    jobs = [obj for obj in contents if obj["Key"].endswith(".json")]
    jobs.sort(key=lambda o: o["LastModified"])  # oldest first
    return jobs


def download_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return json.loads(text)


def upload_json(key: str, data: Dict[str, Any]) -> None:
    body = json.dumps(data, ensure_ascii=False, indent=2)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body.encode("utf-8"))


def move_object(src_key: str, dst_key: str) -> None:
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)


# =============================
# MEDIA PIPE / JOHANSSON LOGIC
# =============================

# Approximate Johansson points using subset of Pose landmarks
JOHANSSON_LANDMARK_IDXS = [
    0,   # nose (หัว)
    11,  # L shoulder
    12,  # R shoulder
    13,  # L elbow
    14,  # R elbow
    15,  # L wrist
    16,  # R wrist
    23,  # L hip
    24,  # R hip
    25,  # L knee
    26,  # R knee
    27,  # L ankle
    28,  # R ankle
]


def render_johansson_dots(
    input_path: str,
    output_path: str,
    dot_radius: int = 5,
) -> None:
    """
    Read input video, run MediaPipe Pose, and render Johansson-style dots.
    Output video has black background + white dots at key joints.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Convert to RGB for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            # Start from black frame
            black = np.zeros_like(frame)

            if result.pose_landmarks:
                h, w, _ = black.shape
                landmarks = result.pose_landmarks.landmark

                for idx in JOHANSSON_LANDMARK_IDXS:
                    if idx >= len(landmarks):
                        continue
                    lm = landmarks[idx]
                    if lm.visibility < 0.5:
                        continue
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            black,
                            (cx, cy),
                            dot_radius,
                            (255, 255, 255),
                            thickness=-1,
                            lineType=cv2.LINE_AA,
                        )

            out.write(black)

    finally:
        cap.release()
        out.release()
        pose.close()

    logger.info(f"Rendered Johansson dots to {output_path}")


# =============================
# JOB PROCESSING
# =============================

def process_single_job(src_key: str) -> None:
    """
    Take one pending job JSON (in jobs/pending),
    move to processing, run overlay, update status.
    """
    logger.info(f"Claiming job: {src_key}")
    job_pending = download_json(src_key)

    job_id = job_pending.get("job_id")
    if not job_id:
        # Derive from filename
        base_name = os.path.basename(src_key)
        job_id = base_name.replace(".json", "")

    # Keys
    pending_key = src_key
    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    failed_key = f"{FAILED_PREFIX}/{job_id}.json"

    # Video keys (ตาม convention เดิม)
    input_key = job_pending.get(
        "input_key",
        f"{JOBS_PREFIX}/{job_id}/input/input.mp4",
    )
    overlay_key = job_pending.get(
        "overlay_key",
        f"{JOBS_PREFIX}/{job_id}/output/dots.mp4",
    )

    # Dot params
    dot_radius = int(job_pending.get("dot_radius", 5))

    # -----------------------
    # Move to processing
    # -----------------------
    logger.info(f"Moving job to processing: {processing_key}")
    move_object(pending_key, processing_key)

    job = job_pending
    job["status"] = "processing"
    job["progress"] = 5
    job["message"] = "Downloading input video"
    job["job_id"] = job_id
    job["input_key"] = input_key
    job["overlay_key"] = overlay_key

    upload_json(processing_key, job)

    # -----------------------
    # Download video
    # -----------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        local_input = os.path.join(tmpdir, "input.mp4")
        local_output = os.path.join(tmpdir, "dots.mp4")

        try:
            logger.info(f"Downloading video from s3://{S3_BUCKET}/{input_key}")
            s3.download_file(S3_BUCKET, input_key, local_input)

            job["status"] = "processing"
            job["progress"] = 30
            job["message"] = "Running MediaPipe Pose"
            upload_json(processing_key, job)

            # -----------------------
            # Run MediaPipe + render
            # -----------------------
            render_johansson_dots(local_input, local_output, dot_radius=dot_radius)

            job["status"] = "processing"
            job["progress"] = 80
            job["message"] = "Uploading overlay"
            upload_json(processing_key, job)

            # -----------------------
            # Upload result
            # -----------------------
            logger.info(f"Uploading overlay to s3://{S3_BUCKET}/{overlay_key}")
            s3.upload_file(local_output, S3_BUCKET, overlay_key)

            job["status"] = "done"
            job["progress"] = 100
            job["message"] = "Completed"
            job["outputs"] = {"overlay_key": overlay_key}
            upload_json(processing_key, job)

            logger.info(f"Job {job_id} completed successfully.")

        except Exception as e:
            logger.exception(f"Job {job_id} failed: {e}")
            job["status"] = "failed"
            job["message"] = f"{type(e).__name__}: {e}"
            job["progress"] = 100
            upload_json(processing_key, job)

            # move JSON -> failed
            move_object(processing_key, failed_key)
            logger.info(f"Job {job_id} moved to failed.")
            return

        # (optional) ถ้าอยาก move processing -> completed แยก folder
        # ตอนนี้ให้ status = done อยู่ใน processing ตามที่ UI ใช้งานอยู่


def main_loop():
    logger.info("AI People Reader worker started.")
    logger.info(f"Region={AWS_REGION} Bucket={S3_BUCKET}")

    while True:
        try:
            jobs = list_pending_jobs()
            if not jobs:
                logger.info("No pending jobs. Sleeping...")
                time.sleep(JOB_POLL_INTERVAL)
                continue

            # Process first job in queue
            job_obj = jobs[0]
            src_key = job_obj["Key"]
            try:
                process_single_job(src_key)
            except Exception as e:
                logger.exception(f"Unexpected error while processing {src_key}: {e}")

        except Exception as outer:
            logger.exception(f"Worker main loop error: {outer}")
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
