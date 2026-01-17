# src/worker.py  — AI People Reader worker (dots / clear / skeleton, NO reports)

import os
import json
import time
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# heavy libs
import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception:  # mediapipe not available
    mp = None  # type: ignore


# --------------------------------------------------------------------
# Config & logging
# --------------------------------------------------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("Starting worker…")
logger.info("Bucket : %s", AWS_BUCKET)
logger.info("Region : %s", AWS_REGION)
logger.info("Poll   : %s sec", POLL_INTERVAL)

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> Dict[str, Any]:
    resp = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = resp["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_delete_key(key: str) -> None:
    try:
        s3.delete_object(Bucket=AWS_BUCKET, Key=key)
    except ClientError as e:
        logger.warning("Delete failed for %s: %s", key, e)


def download_s3_to_temp(key: str) -> str:
    """Download S3 object to a temp file and return path."""
    resp = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = resp["Body"].read()
    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return tmp_path


def upload_file_to_s3(path: str, key: str) -> None:
    with open(path, "rb") as f:
        s3.put_object(
            Bucket=AWS_BUCKET,
            Key=key,
            Body=f,
            ContentType="video/mp4",
        )


def find_one_pending_job_key() -> Optional[str]:
    """ค้นหา pending job 1 ตัว (ไฟล์ .json)"""
    try:
        resp = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=JOBS_PENDING_PREFIX,
        )
    except ClientError as e:
        logger.error("list_objects_v2 error: %s", e)
        return None

    contents = resp.get("Contents")
    if not contents:
        return None

    # เอา key ที่เล็กสุด (เก่าที่สุด)
    json_keys = [
        obj["Key"] for obj in contents if obj["Key"].endswith(".json")
    ]
    if not json_keys:
        return None

    json_keys.sort()
    return json_keys[0]


# --------------------------------------------------------------------
# Video processing functions
# --------------------------------------------------------------------
def _ensure_pose():
    if mp is None:
        raise RuntimeError("mediapipe is not installed in worker environment")
    return mp.solutions.pose.Pose  # type: ignore


def process_video_dots(input_path: str, output_path: str) -> None:
    """สร้างวิดีโอ Johansson dots (พื้นดำ จุดขาวตาม joint)"""
    Pose = _ensure_pose()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    landmark_indices = [
        11, 12,  # shoulders
        13, 14,  # elbows
        15, 16,  # wrists
        23, 24,  # hips
        25, 26,  # knees
        27, 28,  # ankles
    ]

    with Pose(
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
            results = pose.process(rgb)

            dots = np.zeros_like(frame)

            if results.pose_landmarks:
                h_, w_, _ = frame.shape
                for idx in landmark_indices:
                    lm = results.pose_landmarks.landmark[idx]
                    x = int(lm.x * w_)
                    y = int(lm.y * h_)
                    if 0 <= x < w_ and 0 <= y < h_:
                        cv2.circle(dots, (x, y), 5, (255, 255, 255), -1)

            out.write(dots)

    cap.release()
    out.release()


def process_video_clear(input_path: str, output_path: str) -> None:
    """โหมด clear = copy วิดีโอเฉย ๆ"""
    import shutil

    shutil.copyfile(input_path, output_path)


def process_video_skeleton(input_path: str, output_path: str) -> None:
    """วาด skeleton แบบง่ายบนพื้นดำ"""
    Pose = _ensure_pose()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # connections (เหมือน mediapipe pose)
    connections = [
        (11, 12),  # shoulders
        (11, 23),  # L shoulder-hip
        (12, 24),  # R shoulder-hip
        (23, 24),  # hips
        (11, 13), (13, 15),  # left arm
        (12, 14), (14, 16),  # right arm
        (23, 25), (25, 27),  # left leg
        (24, 26), (26, 28),  # right leg
    ]

    with Pose(
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
            results = pose.process(rgb)

            canvas = np.zeros_like(frame)

            if results.pose_landmarks:
                h_, w_, _ = frame.shape
                pts = []
                for lm in results.pose_landmarks.landmark:
                    pts.append((int(lm.x * w_), int(lm.y * h_)))

                # draw lines
                for i, j in connections:
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 255), 3)

                # draw joints
                for x, y in pts:
                    if 0 <= x < w_ and 0 <= y < h_:
                        cv2.circle(canvas, (x, y), 4, (255, 255, 255), -1)

            out.write(canvas)

    cap.release()
    out.release()


def run_job(job_key: str) -> None:
    """อ่าน pending job แล้วประมวลผล"""
    basename = os.path.basename(job_key)
    processing_key = f"{JOBS_PROCESSING_PREFIX}{basename}"
    finished_key = f"{JOBS_FINISHED_PREFIX}{basename}"
    failed_key = f"{JOBS_FAILED_PREFIX}{basename}"

    job = s3_get_json(job_key)
    logger.info("Processing job %s", job.get("job_id"))

    # move -> processing
    job["status"] = "processing"
    job["updated_at"] = utc_now_iso()
    s3_put_json(processing_key, job)
    s3_delete_key(job_key)

    input_key = job["input_key"]
    output_key = job["output_key"]
    mode = job.get("mode", "dots")

    tmp_input = download_s3_to_temp(input_key)
    fd, tmp_output = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)  # will use path only

    try:
        if mode == "dots":
            process_video_dots(tmp_input, tmp_output)
        elif mode == "clear":
            process_video_clear(tmp_input, tmp_output)
        elif mode == "skeleton":
            process_video_skeleton(tmp_input, tmp_output)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        upload_file_to_s3(tmp_output, output_key)

        job["status"] = "finished"
        job["error"] = None
        job["updated_at"] = utc_now_iso()
        s3_put_json(finished_key, job)
        s3_delete_key(processing_key)

        logger.info("Job %s finished OK", job.get("job_id"))

    except Exception as e:
        logger.exception("Job %s FAILED: %s", job.get("job_id"), e)
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = utc_now_iso()
        s3_put_json(failed_key, job)
        s3_delete_key(processing_key)

    finally:
        try:
            os.remove(tmp_input)
        except OSError:
            pass
        try:
            os.remove(tmp_output)
        except OSError:
            pass


def main_loop() -> None:
    logger.info("Worker main loop started")
    while True:
        try:
            job_key = find_one_pending_job_key()
            if not job_key:
                time.sleep(POLL_INTERVAL)
                continue

            logger.info("Found pending job JSON: %s", job_key)
            run_job(job_key)

        except Exception as e:
            logger.exception("Unexpected error in main loop: %s", e)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
