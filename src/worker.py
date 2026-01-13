# worker.py — background worker สำหรับ S3 jobs

import os
import json
import time
import logging

import boto3
from botocore.exceptions import ClientError

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# -------------------------------------------------
# Env
# -------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET") or os.getenv("AWS_BUCKET")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

PENDING_PREFIX = "jobs/pending/"
OUTPUT_PREFIX = "jobs/output/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET (or AWS_BUCKET) env var is required")

s3 = boto3.client("s3", region_name=AWS_REGION)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def list_pending_jobs():
    """คืน list ของ key ไฟล์ .json ใน jobs/pending/"""
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=PENDING_PREFIX)
    except ClientError as e:
        logging.error("Error listing pending jobs: %s", e)
        return []

    keys = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".json") and not key.endswith("/"):
            keys.append(key)
    return keys


def load_json(key: str):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return json.loads(data)


def save_json(data: dict, key: str):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def move_object(src_key: str, dst_key: str):
    """ย้ายไฟล์ภายใน bucket เดียวกัน (copy + delete)"""
    logging.info("Moving %s -> %s", src_key, dst_key)
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)


def copy_video(src_key: str, dst_key: str):
    """
    คัดลอกไฟล์วิดีโอจาก src_key -> dst_key ใน bucket เดียวกัน
    ***สำคัญ: ห้ามมี s3:// ใน CopySource***
    """
    logging.info("Copying video %s -> %s", src_key, dst_key)
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )

# -------------------------------------------------
# Job processing
# -------------------------------------------------
def process_job(job_key: str):
    """
    Dummy processor (เวอร์ชันแรก): 
    - อ่าน job JSON
    - copy video ไปที่ jobs/output/<job_id>/result.mp4
    - เขียน JSON ผลลัพธ์ที่ jobs/output/<job_id>.json
    - ย้าย job JSON เดิมไป jobs/done/...
    """
    job = load_json(job_key)
    job_id = job.get("job_id") or os.path.basename(job_key).replace(".json", "")
    mode = job.get("mode", "dots")
    video_key = job.get("video_key")

    logging.info("Processing job %s (key=%s, mode=%s)", job_id, job_key, mode)
    logging.info("video_key from JSON = %s", video_key)

    if not video_key:
        raise ValueError("Job JSON ไม่มี field 'video_key'")

    # ตรวจว่าไฟล์วิดีโอต้นทางมีจริงไหม
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=video_key)
    except ClientError as e:
        logging.error("Source video not found: %s", e)
        raise

    # สร้าง path ปลายทางสำหรับวิดีโอที่ประมวลผลแล้ว
    result_video_key = f"{OUTPUT_PREFIX}{job_id}/result.mp4"

    # ตอนนี้ทำแค่ copy video (ยังไม่ทำ dot processing)
    copy_video(video_key, result_video_key)

    # เขียนผลลัพธ์ JSON
    output_json_key = f"{OUTPUT_PREFIX}{job_id}.json"
    result = {
        "status": "done",
        "job_id": job_id,
        "mode": mode,
        "video_key": result_video_key,
        "video_url": f"s3://{S3_BUCKET}/{result_video_key}",
    }
    save_json(result, output_json_key)

    # ย้าย job JSON ไปโฟลเดอร์ done
    done_key = f"{DONE_PREFIX}{os.path.basename(job_key)}"
    move_object(job_key, done_key)

    logging.info(
        "Job %s finished. Output JSON at s3://%s/%s",
        job_id,
        S3_BUCKET,
        output_json_key,
    )

# -------------------------------------------------
# Main loop
# -------------------------------------------------
def main():
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
                process_job(job_key)
            except Exception as e:
                logging.exception("Job failed for %s: %s", job_key, e)
                # ย้าย job json ไป failed
                failed_key = f"{FAILED_PREFIX}{os.path.basename(job_key)}"
                try:
                    move_object(job_key, failed_key)
                except Exception:
                    logging.exception(
                        "Also failed to move %s to failed/; leaving in pending/",
                        job_key,
                    )

        # จบรอบหนึ่งแล้ววน loop ต่อ (ไม่ต้อง sleep เพิ่ม)


if __name__ == "__main__":
    main()
