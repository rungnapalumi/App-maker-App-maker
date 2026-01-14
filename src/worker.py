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
        # เอาเฉพาะไฟล์ .json ไม่เอาโฟลเดอร์
        if key.endswith(".json") and not key.endswith("/"):
            keys.append(key)
    return keys


def load_json(key: str):
    """Download and parse JSON from S3"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return json.loads(data)


def save_json(data: dict, key: str):
    """Upload JSON to S3"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def move_object(src_key: str, dst_key: str):
    """Move object inside the same bucket (copy + delete)"""
    logging.info("Moving %s -> %s", src_key, dst_key)
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)


def copy_video(src_key: str, dst_key: str):
    """
    คัดลอกไฟล์วิดีโอจาก src_key -> dst_key ภายใน bucket เดียวกัน
    *** จุดนี้แหละที่เมื่อก่อนพัง เพราะส่ง CopySource ไม่ครบ bucket ***
    """
    logging.info(
        "Copying video s3://%s/%s -> s3://%s/%s",
        S3_BUCKET,
        src_key,
        S3_BUCKET,
        dst_key,
    )
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
    อ่าน job JSON จาก pending, คัดลอกไฟล์วิดีโอไปโฟลเดอร์ output,
    เขียนไฟล์ผลลัพธ์ แล้วย้าย JSON ไป done หรือ failed
    """
    job = load_json(job_key)
    job_id = job.get("job_id") or os.path.basename(job_key).replace(".json", "")
    mode = job.get("mode", "dots")
    video_key = job.get("video_key")

    logging.info("Processing job %s (key=%s)", job_id, job_key)
    logging.info("Mode=%s video_key=%s", mode, video_key)

    if not video_key:
        raise RuntimeError(f"Job {job_id} has no 'video_key' in JSON")

    # ที่เก็บวิดีโอผลลัพธ์ (ตอนนี้ยังแค่ copy เฉย ๆ)
    result_video_key = f"{OUTPUT_PREFIX}{job_id}/result.mp4"

    # 1) คัดลอกวิดีโอจาก pending/input -> output
    copy_video(video_key, result_video_key)

    # 2) เขียนผลลัพธ์ JSON ง่าย ๆ ไว้ก่อน
    result = {
        "status": "done",
        "job_id": job_id,
        "mode": mode,
        "input_video_key": video_key,
        "output_video_key": result_video_key,
    }

    output_json_key = f"{OUTPUT_PREFIX}{job_id}.json"
    save_json(result, output_json_key)

    # 3) ย้าย job JSON จาก pending -> done
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

        for job_key in pending_jobs:
            try:
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

        # เสร็จหนึ่งรอบแล้ว loop ต่อเลย


if __name__ == "__main__":
    main()
