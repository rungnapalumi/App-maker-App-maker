import os
import json
import time
import logging

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
# Helper functions
# -------------------------------------------------
def list_pending_jobs():
    """List pending job JSON files in jobs/pending/"""
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=PENDING_PREFIX)
    except ClientError as e:
        logging.error("Error listing pending jobs: %s", e)
        return []

    keys = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        # เราเอาเฉพาะไฟล์ .json ไม่เอาโฟลเดอร์
        if key.endswith(".json") and not key.endswith("/"):
            keys.append(key)
    return keys


def load_json(key):
    """Download and parse JSON from S3"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return json.loads(data)


def save_json(data, key):
    """Upload JSON to S3"""
    body = json.dumps(data).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def move_object(src_key, dst_key):
    """Move object inside the same bucket (copy + delete)"""
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)


# -------------------------------------------------
# Job processing
# -------------------------------------------------
def process_job(job_key: str):
    """
    ตอนนี้ยังไม่ทำ video processing จริง ๆ
    แค่เปลี่ยนสถานะเป็น done แล้วเขียนผลลัพธ์ไปที่ jobs/output/
    """
    job = load_json(job_key)
    job_id = job.get("job_id") or os.path.basename(job_key).replace(".json", "")

    logging.info("Processing job %s (key=%s)", job_id, job_key)

    # ตรงนี้เดี๋ยวอนาคตเราค่อยเสียบ logic ทำ dot / skeleton ฯลฯ
    # ตอนนี้คือ dummy processor: mark status=done
    result = {
        "status": "done",
        "job_id": job_id,
    }

    # เขียนผลลัพธ์ไปที่ jobs/output/{job_id}.json
    output_key = f"{OUTPUT_PREFIX}{job_id}.json"
    save_json(result, output_key)

    # ย้าย job เดิมจาก pending -> done
    done_key = f"{DONE_PREFIX}{os.path.basename(job_key)}"
    move_object(job_key, done_key)

    logging.info(
        "Job %s finished. Output JSON at s3://%s/%s",
        job_id,
        S3_BUCKET,
        output_key,
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

        for job_key in pending_jobs:
            try:
                process_job(job_key)
            except Exception as e:
                logging.exception("Job failed for %s: %s", job_key, e)
                # ย้ายไฟล์ job JSON ไปโฟลเดอร์ failed แบบเดิม
                failed_key = f"{FAILED_PREFIX}{os.path.basename(job_key)}"
                try:
                    move_object(job_key, failed_key)
                except Exception:
                    logging.exception(
                        "Also failed to move %s to failed/; leaving in pending/",
                        job_key,
                    )

        # เสร็จรอบนึงแล้วกลับไป loop ต่อทันที (ไม่ต้อง sleep เพิ่ม)


if __name__ == "__main__":
    main()
