"""
Background worker for AI People Reader - dot motion video jobs.

- Polls S3 for jobs in jobs/pending/*.json
- Downloads video from S3 (jobs/pending/{job_id}/input/input.mp4)
- Runs pose detection and renders Johansson-style white dots
- Uploads:
    - jobs/output/{job_id}/output.mp4   (processed video)
    - jobs/output/{job_id}.json        (status JSON: {"status": "done", ...})
- On error uploads:
    - jobs/failed/{job_id}.json        (status JSON: {"status": "failed", "error": ...})

Job JSON structure (created by the frontend app) is expected to be:

{
  "job_id": "20260113_133856__02d9f4",
  "mode": "dots",
  "video_key": "jobs/pending/20260113_133856__02d9f4/input/input.mp4",
  "created_at": "...",
  "status": "pending"
}

"""

import os
import json
import time
import tempfile
import logging
from typing import Optional, Dict

import boto3
import botocore
import cv2
import numpy as np
import mediapipe as mp  # ✅ ห้ามใช้ mediapipe.python.solutions แล้ว


# -----------------------------------------------------------------------------
# Config & logging
# -----------------------------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

PENDING_PREFIX = "jobs/pending/"
OUTPUT_PREFIX = "jobs/output/"
FAILED_PREFIX = "jobs/failed/"
DONE_PREFIX = "jobs/done/"

INPUT_VIDEO_RELATIVE_PATH = "input/input.mp4"  # ภายในโฟลเดอร์ของ job ใน pending/

if not AWS_BUCKET:
    raise RuntimeError("AWS_BUCKET or S3_BUCKET must be set in environment variables")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

logger.info(
    "Worker starting. bucket=%s region=%s poll_interval=%ss",
    AWS_BUCKET,
    AWS_REGION,
    JOB_POLL_INTERVAL,
)

s3 = boto3.client("s3", region_name=AWS_REGION)

# Mediapipe pose model (ใช้ global แค่ตัวเดียว ลด overhead)
mp_pose = mp.solutions.pose


# -----------------------------------------------------------------------------
# S3 helper functions
# -----------------------------------------------------------------------------

def list_pending_job_json() -> Optional[str]:
    """
    หา job JSON ตัวแรกใน jobs/pending/ ที่ลงท้ายด้วย .json
    คืนค่าเป็น key (เช่น 'jobs/pending/20260113_133856__02d9f4.json')
    ถ้าไม่เจอ คืน None
    """
    resp = s3.list_objects_v2(
        Bucket=AWS_BUCKET,
        Prefix=PENDING_PREFIX,
        MaxKeys=50,
    )
    contents = resp.get("Contents", [])
    for obj in contents:
        key = obj["Key"]
        if key.endswith(".json"):
            return key
    return None


def download_s3_to_file(key: str, local_path: str) -> None:
    logger.info("Downloading s3://%s/%s -> %s", AWS_BUCKET, key, local_path)
    s3.download_file(AWS_BUCKET, key, local_path)


def upload_file_to_s3(local_path: str, key: str, content_type: Optional[str] = None) -> None:
    logger.info("Uploading %s -> s3://%s/%s", local_path, AWS_BUCKET, key)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    s3.upload_file(local_path, AWS_BUCKET, key, ExtraArgs=extra)


def move_s3_object(src_key: str, dest_key: str) -> None:
    """
    ย้ายไฟล์ใน S3 โดย copy แล้ว delete
    """
    logger.info("Moving s3://%s/%s -> s3://%s/%s", AWS_BUCKET, src_key, AWS_BUCKET, dest_key)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": src_key},
        Key=dest_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)


# -----------------------------------------------------------------------------
# Video processing: Dots renderer
# -----------------------------------------------------------------------------

def process_video_to_dots(
    input_path: str,
    output_path: str,
    dot_size: int = 2,
) -> None:
    """
    แปลงวิดีโอต้นฉบับเป็น dot motion video แบบ Johansson
    (ไม่มีเสียง เพื่อลด dependency moviepy ใน worker)
    """
    logger.info("Processing video to dots: %s -> %s", input_path, output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            # พื้นหลังดำ
            output_frame = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            output_frame,
                            (cx, cy),
                            radius=dot_size,
                            color=(255, 255, 255),  # white
                            thickness=-1,
                        )

            out.write(output_frame)
            frame_idx += 1

            if total_frames and frame_idx % 50 == 0:
                logger.info("Processed %d / %d frames", frame_idx, total_frames)

    cap.release()
    out.release()
    logger.info("Dot video created: %s", output_path)


# -----------------------------------------------------------------------------
# Job processing
# -----------------------------------------------------------------------------

def load_job_json(local_path: str) -> Dict:
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_status_json(
    local_path: str,
    status: str,
    job_id: str,
    extra: Optional[Dict] = None,
) -> None:
    data = {"status": status, "job_id": job_id}
    if extra:
        data.update(extra)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_single_job(job_json_key: str) -> None:
    """
    Process one job described by JSON at job_json_key.
    """
    job_id = os.path.splitext(os.path.basename(job_json_key))[0]
    logger.info("Processing job_id=%s (json key=%s)", job_id, job_json_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_job_json = os.path.join(tmpdir, "job.json")
        download_s3_to_file(job_json_key, local_job_json)

        job = load_job_json(local_job_json)
        video_key = job.get("video_key")
        mode = job.get("mode", "dots")
        dot_size = int(job.get("dot_size", 2))

        if not video_key:
            raise RuntimeError("Job JSON missing 'video_key'")

        logger.info("Job %s: mode=%s video_key=%s", job_id, mode, video_key)

        # Download input video
        local_input_video = os.path.join(tmpdir, "input.mp4")
        download_s3_to_file(video_key, local_input_video)

        # Process
        local_output_video = os.path.join(tmpdir, "output.mp4")

        if mode == "dots":
            process_video_to_dots(local_input_video, local_output_video, dot_size=dot_size)
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        # Upload outputs
        output_video_key = f"{OUTPUT_PREFIX}{job_id}/output.mp4"
        upload_file_to_s3(local_output_video, output_video_key, content_type="video/mp4")

        # Status JSON (done)
        local_status_json = os.path.join(tmpdir, "status.json")
        write_status_json(
            local_status_json,
            status="done",
            job_id=job_id,
            extra={"video_key": output_video_key, "mode": mode},
        )
        output_status_key = f"{OUTPUT_PREFIX}{job_id}.json"
        upload_file_to_s3(local_status_json, output_status_key, content_type="application/json")

        # Move job JSON จาก pending -> done (เก็บเป็น log)
        done_job_key = f"{DONE_PREFIX}{job_id}.json"
        move_s3_object(job_json_key, done_job_key)

        logger.info("Job %s finished successfully", job_id)


def handle_job_failure(job_json_key: str, error: Exception) -> None:
    """
    ถ้า process job ล้มเหลว:
      - โหลด job.json (ถ้าดาวน์โหลดได้)
      - เขียน failed JSON ไปที่ jobs/failed/{job_id}.json
      - ย้าย job.json จาก pending -> failed/{job_id}.json (เป็น archive)
    """
    job_id = os.path.splitext(os.path.basename(job_json_key))[0]
    logger.error("Job %s failed: %s", job_id, error, exc_info=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_job_json = os.path.join(tmpdir, "job.json")

        # พยายามโหลด job เดิม (ถ้าดาวน์โหลดไม่ได้ก็ยังสร้าง failed JSON ได้อยู่ดี)
        try:
            download_s3_to_file(job_json_key, local_job_json)
            job = load_job_json(local_job_json)
        except Exception:
            job = {}

        local_failed_json = os.path.join(tmpdir, "failed.json")
        write_status_json(
            local_failed_json,
            status="failed",
            job_id=job_id,
            extra={
                "error": str(error),
                "mode": job.get("mode"),
                "video_key": job.get("video_key"),
            },
        )

        failed_status_key = f"{FAILED_PREFIX}{job_id}.json"
        upload_file_to_s3(local_failed_json, failed_status_key, content_type="application/json")

        # ย้าย job.json จาก pending -> failed/{job_id}.json เพื่อเก็บ log
        failed_job_key = f"{FAILED_PREFIX}{job_id}.json"
        try:
            move_s3_object(job_json_key, failed_job_key)
        except Exception:
            # ถ้าย้ายไม่ได้ (เช่นไฟล์หายไปแล้ว) ก็ข้าม
            logger.warning(
                "Could not move pending job json %s to %s", job_json_key, failed_job_key
            )


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main_loop() -> None:
    while True:
        try:
            job_json_key = list_pending_job_json()
            if not job_json_key:
                logger.info("No pending jobs. Sleeping %s seconds...", JOB_POLL_INTERVAL)
                time.sleep(JOB_POLL_INTERVAL)
                continue

            logger.info("Found pending job JSON: %s", job_json_key)

            try:
                process_single_job(job_json_key)
            except botocore.exceptions.ClientError as e:
                # error จาก S3 (เช่น video ไม่อยู่ใน bucket)
                handle_job_failure(job_json_key, e)
            except Exception as e:
                handle_job_failure(job_json_key, e)

        except Exception as outer:
            # error ระดับ worker เอง (ไม่ใช่ของ job เดียว)
            logger.error("Unexpected error in main loop: %s", outer, exc_info=True)
            # กันไม่ให้ crash loop ทั้งตัว
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
