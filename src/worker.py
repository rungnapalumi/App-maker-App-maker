# src/worker.py

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional

import boto3
import botocore
import cv2
import numpy as np

from config import (
    AWS_BUCKET,
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    FAILED_PREFIX,
    OUTPUT_PREFIX,
    REGION_NAME,
)

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("worker")

# ----------------------------
# S3 client
# ----------------------------
session = boto3.session.Session(region_name=REGION_NAME)
s3 = session.client("s3")


# ----------------------------
# Helper: S3 utils
# ----------------------------
def list_pending_job_key() -> Optional[str]:
    """Return the first pending job json key or None if queue empty."""
    resp = s3.list_objects_v2(
        Bucket=AWS_BUCKET,
        Prefix=PENDING_PREFIX,
        MaxKeys=1,
    )
    contents = resp.get("Contents")
    if not contents:
        return None

    # Only job json files
    for obj in contents:
        key = obj["Key"]
        if key.endswith(".json"):
            return key
    return None


def move_object(key_from: str, key_to: str) -> None:
    """Copy then delete -> emulates 'mv' in S3."""
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": key_from},
        Key=key_to,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=key_from)


def read_job_json(key: str) -> dict:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return json.loads(body)


def write_job_result_json(job_id: str, payload: dict) -> None:
    out_key = f"{OUTPUT_PREFIX}{job_id}/result.json"
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=out_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


# ----------------------------
# Video Processing: Johansson dots
# ----------------------------
def create_johansson_dots(input_path: str, output_path: str) -> None:
    """
    Convert the input video into a Johansson-style moving-dots video.

    Implementation:
      - background subtraction -> เฉพาะส่วนที่ขยับ
      - แปลงเป็น mask, หา contour
      - วาดจุดสีขาวบนพื้นหลังดำตาม contour
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # background-subtractor สำหรับดึงเฉพาะคนที่ขยับ
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ลด noise นิดหน่อย
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        fgmask = back_sub.apply(blur)

        # threshold ให้ได้ mask ชัดๆ
        _, mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # ปรับรูปร่างเล็กน้อย
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # เฟรมดำเปล่า
        dots_frame = np.zeros_like(frame)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:  # ตัดจุดรบกวนเล็กๆ
                continue

            # sample จุดตามเส้น contour
            step = max(1, len(cnt) // 25)  # ประมาณ 25 จุดต่อ contour
            for i in range(0, len(cnt), step):
                x, y = cnt[i][0]
                cv2.circle(dots_frame, (x, y), 4, (255, 255, 255), -1)

        out.write(dots_frame)

    cap.release()
    out.release()


# ----------------------------
# Main job handler
# ----------------------------
def handle_job(job_key: str) -> None:
    """
    Process a single job:
      1) ย้ายจาก pending -> processing
      2) อ่าน job json
      3) โหลด video จาก S3
      4) สร้าง Johansson dots
      5) อัปโหลดผลที่ output/
    """
    job_id = Path(job_key).stem
    processing_key = f"{PROCESSING_PREFIX}{job_id}.json"
    failed_key = f"{FAILED_PREFIX}{job_id}.json"

    logger.info(f"Claim job {job_id}: {job_key} -> {processing_key}")
    move_object(job_key, processing_key)

    try:
        job_data = read_job_json(processing_key)
        # รองรับได้หลายชื่อ field เผื่อฝั่ง web ใช้ไม่เหมือนกัน
        video_key = (
            job_data.get("video_key")
            or job_data.get("input_key")
            or job_data.get("source_key")
        )

        if not video_key:
            raise ValueError("Job JSON ไม่มี field video_key / input_key / source_key")

        mode = job_data.get("mode", "dots")

        logger.info(f"Job {job_id}: video_key={video_key}, mode={mode}")

        # โหลดวิดีโอลง /tmp
        local_input = f"/tmp/{job_id}_input.mp4"
        local_output = f"/tmp/{job_id}_dots.mp4"

        logger.info(f"Downloading s3://{AWS_BUCKET}/{video_key} -> {local_input}")
        Path(local_input).parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(AWS_BUCKET, video_key, local_input)

        if mode == "dots":
            logger.info(f"Creating Johansson dots for job {job_id}")
            create_johansson_dots(local_input, local_output)
            output_name = "dots.mp4"
        else:
            # ถ้าไม่รู้ mode ก็แค่ก็อปไฟล์เดิมให้ก่อน
            logger.warning(
                f"Unknown mode '{mode}' for job {job_id}, copying input -> output"
            )
            import shutil

            shutil.copy(local_input, local_output)
            output_name = "output.mp4"

        # อัปโหลดผลไปที่ jobs/output/<job_id>/dots.mp4
        output_key = f"{OUTPUT_PREFIX}{job_id}/{output_name}"
        logger.info(f"Uploading result to s3://{AWS_BUCKET}/{output_key}")
        s3.upload_file(local_output, AWS_BUCKET, output_key)

        # เขียน result.json เผื่อ frontend เอาไปใช้
        write_job_result_json(
            job_id,
            {
                "job_id": job_id,
                "status": "completed",
                "mode": mode,
                "output_key": output_key,
            },
        )

        # ลบ job json ออกจาก processing (ถือว่าเสร็จแล้ว)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)
        logger.info(f"Job {job_id} done.")

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        # ย้าย job ไปโฟลเดอร์ failed ไว้ดูทีหลัง
        try:
            move_object(processing_key, failed_key)
        except botocore.exceptions.ClientError:
            # ถ้าย้ายไม่ได้เพราะไฟล์หายไปแล้ว ก็ข้าม
            logger.warning("Could not move job json to FAILED prefix.")


# ----------------------------
# Worker loop
# ----------------------------
def main_loop(poll_interval: float = 5.0) -> None:
    logger.info(
        f"Worker starting. Bucket={AWS_BUCKET}, "
        f"region={REGION_NAME}, prefixes: {PENDING_PREFIX}, {OUTPUT_PREFIX}"
    )

    while True:
        try:
            job_key = list_pending_job_key()
            if not job_key:
                logger.info("No pending jobs. Sleeping...")
                time.sleep(poll_interval)
                continue

            handle_job(job_key)

        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            time.sleep(poll_interval)


if __name__ == "__main__":
    main_loop()
