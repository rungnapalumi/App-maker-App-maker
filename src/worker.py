# src/worker.py
"""
Background worker สำหรับประมวลผลวิดีโอบน S3

โครงหลัก:
- Poll หา jobs/pending/*.json จาก S3
- อ่าน job JSON → ย้ายไป processing → ประมวลผลตาม mode
- เซฟไฟล์ผลลัพธ์ (result.mp4 หรือไฟล์อื่น ๆ) ไปที่ jobs/output/<job_id>/
- อัปเดต job JSON ไปที่ jobs/finished/ หรือ jobs/failed/

รองรับ mode ตอนนี้:
- "dots"      → วิดีโอจุดแบบ Johansson จาก Mediapipe Pose
- "clear"     → copy วิดีโอต้นฉบับเป็น result.mp4
- "skeleton"  → วิดีโอเส้นโครงกระดูก (ใช้ Pose connections)
(ภายหลังจะเพิ่ม "presentation" ได้)
"""

import json
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# พยายาม import stack สำหรับวิดีโอ
try:
    import cv2
except Exception:  # ImportError หรือ error อื่น ๆ
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    import numpy as np
except Exception:
    np = None


# ---------------------------------------------------------------------
# Config & logger
# ---------------------------------------------------------------------

S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

if not S3_BUCKET:
    raise RuntimeError("Missing S3_BUCKET (or AWS_BUCKET) environment variable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s - %(message)s",
)
logger = logging.getLogger("worker")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def copy_video_in_s3(input_key: str, output_key: str) -> None:
    """
    ใช้ S3 copy ภายใน bucket เดียวกัน (ไม่ต้องโหลดขึ้นลง)
    """
    logger.info(f"Copy video in S3: {input_key} → {output_key}")
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": input_key},
        Key=output_key,
        ContentType="video/mp4",
    )


def ensure_cv_stack():
    """
    ถ้า stack วิดีโอ import ไม่ได้ ให้ error ชัด ๆ
    """
    if cv2 is None or mp is None or np is None:
        raise RuntimeError(
            "OpenCV / Mediapipe / NumPy not available. "
            "Please check requirements.txt and that packages are installed."
        )


# ---------------------------------------------------------------------
# Video processing: dots / skeleton
# ---------------------------------------------------------------------


def render_pose_video(
    input_path: str,
    output_path: str,
    style: str = "dots",
) -> None:
    """
    ใช้ Mediapipe Pose ทำวิดีโอ:
    - style="dots"     → เฟรมดำ + จุดขาวตาม joint
    - style="skeleton" → เฟรมดำ + จุด + เส้นเชื่อม joint
    """
    ensure_cv_stack()

    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    logger.info(
        f"Processing video with mediapipe pose: style={style}, fps={fps}, size=({w}x{h})"
    )

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
            result = pose.process(rgb)

            dots_frame = np.zeros_like(frame)  # background ดำ
            if result.pose_landmarks:
                h_f, w_f, _ = frame.shape

                # วาดจุด
                for lm in result.pose_landmarks.landmark:
                    if lm.visibility < 0.5:
                        continue
                    x = int(lm.x * w_f)
                    y = int(lm.y * h_f)
                    cv2.circle(dots_frame, (x, y), 6, (255, 255, 255), -1)

                # วาดเส้น ถ้า style = skeleton
                if style == "skeleton":
                    lm_list = result.pose_landmarks.landmark
                    for start_idx, end_idx in mp_pose.POSE_CONNECTIONS:
                        s = lm_list[start_idx]
                        e = lm_list[end_idx]
                        if s.visibility < 0.5 or e.visibility < 0.5:
                            continue
                        x1, y1 = int(s.x * w_f), int(s.y * h_f)
                        x2, y2 = int(e.x * w_f), int(e.y * h_f)
                        cv2.line(dots_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            out.write(dots_frame)

    cap.release()
    out.release()
    logger.info("Finished render_pose_video")


def process_dots_job(job: Dict[str, Any], input_key: str, output_key: str) -> None:
    """
    ดึงไฟล์จาก S3 → ทำ dot → อัปโหลดกลับ
    """
    logger.info(f"[{job['job_id']}] process_dots_job")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "result.mp4")

        # download input
        s3.download_file(S3_BUCKET, input_key, input_path)

        # process
        render_pose_video(input_path, output_path, style="dots")

        # upload output
        s3.upload_file(
            output_path,
            S3_BUCKET,
            output_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )


def process_skeleton_job(job: Dict[str, Any], input_key: str, output_key: str) -> None:
    """
    ดึงไฟล์จาก S3 → ทำ skeleton → อัปโหลดกลับ
    """
    logger.info(f"[{job['job_id']}] process_skeleton_job")
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "result.mp4")

        s3.download_file(S3_BUCKET, input_key, input_path)

        render_pose_video(input_path, output_path, style="skeleton")

        s3.upload_file(
            output_path,
            S3_BUCKET,
            output_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )


def process_clear_job(job: Dict[str, Any], input_key: str, output_key: str) -> None:
    """
    โหมด clear: ตอนนี้ copy วิดีโอต้นฉบับเป็น result.mp4 เฉย ๆ
    """
    logger.info(f"[{job['job_id']}] process_clear_job (copy video)")
    copy_video_in_s3(input_key=input_key, output_key=output_key)


# ---------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------


def find_one_pending_job_key() -> Optional[str]:
    """
    หา pending job 1 อันจาก S3 (jobs/pending/*.json)
    ตอนนี้หยิบตัวแรกที่เจอ
    """
    resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=JOBS_PENDING_PREFIX)
    contents = resp.get("Contents")
    if not contents:
        return None

    # เอาเฉพาะ .json
    keys = [o["Key"] for o in contents if o["Key"].endswith(".json")]
    if not keys:
        return None

    # sort ตามชื่อไฟล์ (job_id มี timestamp อยู่แล้ว)
    keys.sort()
    return keys[0]


def process_job(job_key: str) -> None:
    """
    main flow ของการประมวลผล job เดียว
    """
    logger.info(f"Start process_job: {job_key}")

    # อ่าน job JSON
    job = s3_get_json(job_key)
    job_id = job.get("job_id") or os.path.splitext(os.path.basename(job_key))[0]

    # เตรียม key ตามสถานะ
    processing_key = f"{JOBS_PROCESSING_PREFIX}{job_id}.json"
    finished_key = f"{JOBS_FINISHED_PREFIX}{job_id}.json"
    failed_key = f"{JOBS_FAILED_PREFIX}{job_id}.json"

    # เปลี่ยนจาก pending → processing
    job["job_id"] = job_id
    job["status"] = "processing"
    job["updated_at_utc"] = utc_now_iso()

    # เขียน processing แล้วลบ pending
    s3_put_json(processing_key, job)
    s3.delete_object(Bucket=S3_BUCKET, Key=job_key)

    input_key = job.get("input_key")
    output_key = job.get("output_key")
    mode = job.get("mode", "dots")

    if not input_key or not output_key:
        raise ValueError("Job JSON missing 'input_key' or 'output_key'")

    try:
        if mode == "dots":
            process_dots_job(job, input_key, output_key)
        elif mode == "clear":
            process_clear_job(job, input_key, output_key)
        elif mode == "skeleton":
            process_skeleton_job(job, input_key, output_key)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    except Exception as exc:
        # เกิด error → failed
        logger.exception("Error while processing job")
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["traceback"] = traceback.format_exc()
        job["updated_at_utc"] = utc_now_iso()
        s3_put_json(failed_key, job)
        # ลบ processing JSON
        s3.delete_object(Bucket=S3_BUCKET, Key=processing_key)
        return

    # ถ้าทุกอย่างปกติ → finished
    job["status"] = "finished"
    job["error"] = None
    job["updated_at_utc"] = utc_now_iso()
    s3_put_json(finished_key, job)
    s3.delete_object(Bucket=S3_BUCKET, Key=processing_key)
    logger.info(f"[{job_id}] Finished job successfully")


def main_loop() -> None:
    logger.info(f"Worker started. Bucket={S3_BUCKET}, Region={AWS_REGION}")
    while True:
        try:
            job_key = find_one_pending_job_key()
            if not job_key:
                time.sleep(POLL_INTERVAL)
                continue

            logger.info(f"Found pending job: {job_key}")
            process_job(job_key)

        except Exception:
            logger.exception("Unexpected error in main_loop")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
