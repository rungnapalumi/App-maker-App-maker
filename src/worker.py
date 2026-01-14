# src/worker.py
import os
import json
import time
import logging
from typing import List, Dict, Any

import boto3
from botocore.exceptions import ClientError

# -------------------------------------------------
# Basic logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -------------------------------------------------
# Environment
# -------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# ใช้ S3_BUCKET เป็นหลัก ถ้าไม่มีให้ลอง AWS_BUCKET เผื่อไว้
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET")

# หน่วงเวลาระหว่างการวน loop เช็คงาน
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

# โฟลเดอร์หลักใน S3
PENDING_PREFIX = "jobs/pending/"
OUTPUT_PREFIX = "jobs/output/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET (or AWS_BUCKET) env var is required")

# สร้าง S3 client
s3 = boto3.client("s3", region_name=AWS_REGION)


# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def list_pending_jobs() -> List[str]:
    """
    ดึงรายการไฟล์ .json ที่อยู่ใน jobs/pending/
    (เฉพาะไฟล์ metadata ของ job ไม่เอาโฟลเดอร์)
    """
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=PENDING_PREFIX)
    except ClientError as e:
        logging.error("Error listing pending jobs: %s", e)
        return []

    keys: List[str] = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        # เอาเฉพาะไฟล์ .json (ไม่ใช่โฟลเดอร์)
        if key.endswith(".json") and not key.endswith("/"):
            keys.append(key)
    return keys


def load_json(key: str) -> Dict[str, Any]:
    """ดาวน์โหลดและ parse JSON จาก S3"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return json.loads(data)


def save_json(data: Dict[str, Any], key: str) -> None:
    """อัปโหลด JSON ขึ้น S3"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def move_object(src_key: str, dst_key: str) -> None:
    """
    ย้ายไฟล์ใน bucket เดียวกัน (copy + delete)

    *** สำคัญ: CopySource ต้องเป็น dict {"Bucket": ..., "Key": ...}
    ห้ามใช้สตริงแบบ 's3://bucket/key' ไม่งั้นจะเจอ Invalid copy source bucket name
    """
    logging.info("Moving %s -> %s", src_key, dst_key)

    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)


def copy_video(src_key: str, dst_key: str) -> None:
    """
    คัดลอกไฟล์วิดีโอใน bucket เดียวกันจาก src_key -> dst_key
    ตอนนี้ยังไม่ประมวลผล dot จริง แค่ copy ไฟล์ผลลัพธ์ไปให้โหลดได้ก่อน
    """
    logging.info("Copying video %s -> %s", src_key, dst_key)

    # เช่น src_key = "jobs/pending/<job_id>/input/input.mp4"
    # และ dst_key  = "jobs/output/<job_id>/result.mp4"
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )


# -------------------------------------------------
# Job processing
# -------------------------------------------------
def process_job(job_key: str) -> None:
    """
    ประมวลผลงาน 1 job จากไฟล์ JSON ใน jobs/pending/*.json

    เวอร์ชันนี้:
    - อ่าน job metadata
    - copy วิดีโอจาก pending/input/input.mp4 -> output/<job_id>/result.mp4
    - สร้างไฟล์ผลลัพธ์ jobs/output/<job_id>.json
    - ย้ายไฟล์ job เดิมไป jobs/done/
    """
    job = load_json(job_key)
    basename = os.path.basename(job_key)
    job_id = job.get("job_id") or basename.replace(".json", "")

    logging.info("Processing job %s (key=%s)", job_id, job_key)

    video_key = job.get("video_key")
    if not video_key:
        raise ValueError(f"Job {job_id} missing 'video_key' in JSON")

    # กำหนดตำแหน่ง result video ใน S3
    result_video_key = f"{OUTPUT_PREFIX}{job_id}/result.mp4"

    # 1) copy วิดีโอจาก pending -> output
    copy_video(video_key, result_video_key)

    # 2) เขียนผลลัพธ์เป็น JSON (ตอนนี้เป็น dummy: แค่บอกว่า done + path วิดีโอ)
    result = {
        "status": "done",
        "job_id": job_id,
        "mode": job.get("mode", "dots"),
        "video_key": video_key,
        "result_video_key": result_video_key,
    }

    output_json_key = f"{OUTPUT_PREFIX}{job_id}.json"
    save_json(result, output_json_key)

    # 3) ย้าย job เดิมจาก pending -> done
    done_key = f"{DONE_PREFIX}{basename}"
    move_object(job_key, done_key)

    logging.info(
        "Job %s finished. JSON: s3://%s/%s, result video: s3://%s/%s",
        job_id,
        S3_BUCKET,
        output_json_key,
        S3_BUCKET,
        result_video_key,
    )


# -------------------------------------------------
# Main loop
# -------------------------------------------------
def main() -> None:
    logging.info(
        "Worker starting. bucket=%s region=%s poll_interval=%ss",
        S3_BUCKET,
        AWS_REGION,
        POLL_INTERVAL,
    )

    while True:
        pending_jobs = list_pending_jobs()

        if not pending_jobs:
            logging.info("No pending jobs. Sleeping %s seconds...", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)
            continue

        logging.info("Found %d pending job(s)", len(pending_jobs))

        for job_key in pending_jobs:
            try:
                logging.info("Loading JSON from %s", job_key)
                process_job(job_key)
            except Exception as e:
                logging.exception("Job failed for %s: %s", job_key, e)

                # ย้ายไฟล์ job JSON ไปโฟลเดอร์ failed
                failed_key = f"{FAILED_PREFIX}{os.path.basename(job_key)}"
                try:
                    move_object(job_key, failed_key)
                except Exception:
                    logging.exception(
                        "Also failed to move %s to failed/; leaving in pending/",
                        job_key,
                    )

        # จบรอบนึงแล้ววน loop ต่อเลย ไม่ต้อง sleep เพิ่ม


if __name__ == "__main__":
    main()
