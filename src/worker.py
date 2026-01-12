# src/worker.py

import json
import time
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import (
    AWS_BUCKET,
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    FAILED_PREFIX,
    OUTPUT_PREFIX,
    REGION_NAME,
)

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("worker")

# -------------------------------------------------------------------
# S3 client
# -------------------------------------------------------------------
s3 = boto3.client("s3", region_name=REGION_NAME)


# -------------------------------------------------------------------
# Helper: list one pending job
# -------------------------------------------------------------------
def get_one_pending_job() -> Optional[str]:
    """
    Return the key of one pending job JSON in S3, or None if no jobs.
    """
    try:
        resp = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix=PENDING_PREFIX,
            MaxKeys=1,
        )
    except ClientError as e:
        logger.error("Error listing pending jobs: %s", e)
        return None

    contents = resp.get("Contents", [])
    if not contents:
        return None

    # pick first object that looks like a job json
    for obj in contents:
        key = obj["Key"]
        if key.endswith(".json"):
            return key

    return None


# -------------------------------------------------------------------
# Claim job: move from pending -> processing
# -------------------------------------------------------------------
def claim_job(pending_key: str) -> Optional[str]:
    """
    Atomically move a job JSON from pending/ to processing/.
    Returns the new key under processing/ or None on error.
    """
    processing_key = pending_key.replace(PENDING_PREFIX, PROCESSING_PREFIX, 1)

    try:
        # IMPORTANT: CopySource ต้องเป็น dict {Bucket, Key}
        copy_source = {"Bucket": AWS_BUCKET, "Key": pending_key}

        s3.copy_object(
            CopySource=copy_source,
            Bucket=AWS_BUCKET,
            Key=processing_key,
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)

        logger.info("Claimed job: %s -> %s", pending_key, processing_key)
        return processing_key

    except ClientError as e:
        logger.error("Failed to claim job %s: %s", pending_key, e)
        return None


# -------------------------------------------------------------------
# Load / save job JSON
# -------------------------------------------------------------------
def load_job(job_key: str) -> Optional[dict]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=job_key)
        data = obj["Body"].read().decode("utf-8")
        return json.loads(data)
    except ClientError as e:
        logger.error("Failed to load job %s: %s", job_key, e)
        return None


def save_job(job_key: str, job: dict) -> None:
    body = json.dumps(job).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=job_key, Body=body)


# -------------------------------------------------------------------
# Dummy processing (แทนที่ด้วย Johansson / skeleton ภายหลังได้)
# -------------------------------------------------------------------
def process_video(job: dict, job_key: str) -> bool:
    """
    ตรงนี้ตอนนี้ให้ทำงานง่าย ๆ ก่อน:
    - อ่าน field 'input_key'
    - copy วิดีโอจาก input ไปที่ OUTPUT_PREFIX/<job_id>/dots.mp4

    ภายหลังค่อยเปลี่ยนมาเรียกโค้ดทำ Johansson / skeleton จริง
    """
    job_id = job.get("job_id", "unknown")
    input_key = job.get("input_key")

    if not input_key:
        logger.error("Job %s missing input_key", job_id)
        return False

    output_key = os.path.join(OUTPUT_PREFIX, job_id, "dots.mp4")

    try:
        copy_source = {"Bucket": AWS_BUCKET, "Key": input_key}

        s3.copy_object(
            CopySource=copy_source,
            Bucket=AWS_BUCKET,
            Key=output_key,
        )

        logger.info("Processed job %s -> %s", job_id, output_key)
        return True

    except ClientError as e:
        logger.error("Failed to process video for job %s: %s", job_id, e)
        return False


# -------------------------------------------------------------------
# Mark failed (ย้ายไป failed/)
# -------------------------------------------------------------------
def move_to_failed(processing_key: str, job: dict) -> None:
    failed_key = processing_key.replace(PROCESSING_PREFIX, FAILED_PREFIX, 1)
    job["status"] = "failed"

    try:
        # เขียน job JSON ใหม่ไปที่ failed/
        body = json.dumps(job).encode("utf-8")
        s3.put_object(Bucket=AWS_BUCKET, Key=failed_key, Body=body)
        # ลบตัวเดิมใน processing/
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)

        logger.info("Moved job to failed: %s", failed_key)

    except ClientError as e:
        logger.error("Failed to move job to failed: %s", e)


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------
POLL_INTERVAL_SECONDS = 5


def main():
    logger.info("Worker starting. Bucket=%s, region=%s", AWS_BUCKET, REGION_NAME)

    while True:
        pending_key = get_one_pending_job()

        if not pending_key:
            logger.info("No pending jobs. Sleeping %ds...", POLL_INTERVAL_SECONDS)
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        processing_key = claim_job(pending_key)
        if not processing_key:
            # claim ไม่สำเร็จ ลองใหม่รอบหน้า
            time.sleep(1)
            continue

        job = load_job(processing_key)
        if job is None:
            logger.error("Could not load job json for %s", processing_key)
            # move to failed แบบ job ว่าง ๆ
            move_to_failed(processing_key, {"status": "failed", "reason": "load_error"})
            continue

        ok = process_video(job, processing_key)
        job["status"] = "done" if ok else "failed"

        if ok:
            # เขียนผลกลับเข้า processing_key (หรือจะย้ายไป output/ อีกที ก็ได้)
            save_job(processing_key, job)
            logger.info("Job %s done.", job.get("job_id", "?"))
        else:
            move_to_failed(processing_key, job)


if __name__ == "__main__":
    main()
