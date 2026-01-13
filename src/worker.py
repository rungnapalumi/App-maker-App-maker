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


def load_json(key: str) -> dict:
    """Download and parse JSON from S3"""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return json.loads(data)


def save_json(data: dict, key: str) -> None:
    """Upload JSON to S3"""
    body = json.dumps(data).encode("utf-8")
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def move_object(src_key: str, dst_key: str) -> None:
    """
    Move object inside the same bucket (copy + delete).

    IMPORTANT:
    ใช้รูปแบบ dict ให้ S3 ชัดเจนว่า bucket ไหน / key ไหน
    ป้องกัน error: Invalid copy source bucket name
    """
    copy_source = {"Bucket": S3_BUCKET, "Key": src_key}
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource=copy_source,
        Key=dst_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=src_key)


def copy_video(src_key: str, dst_key: str) -> None:
    """
    Copy video from src_key -> dst_key ใน bucket เดียวกัน

    src_key  เช่น jobs/pending/{job_id}__/input/input.mp4
    dst_key  เช่น jobs/output/{job_id}__.mp4  (หรือ path ที่กำหนดเอง)
    """
    logging.info("Copying video from %s to %s", src_key, dst_key)
    copy_source = {"Bucket": S3_BUCKET, "Key": src_key}
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource=copy_source,
        Key=dst_key,
    )


# -------------------------------------------------
# Job processing
# -------------------------------------------------
def process_job(job_key: str) -> None:
    """
    Simple processor:
    - อ่าน job JSON จาก jobs/pending/
    - ถ้ามี video_key ให้ copy วิดีโอไป output
    - เขียนผล JSON ไป jobs/output/{job_id}.json
    - ย้าย job JSON ไป jobs/done/
    """
    job = load_json(job_key)
    job_id = job.get("job_id") or os.path.basename(job_key).replace(".json", "")

    logging.info("Processing job %s (key=%s)", job_id, job_key)

    video_key = job.get("video_key")

    result_video_key = None
    if video_key:
        # พยายาม map จาก pending -> output แบบ automatic
        # ตัวอย่าง:
        #   jobs/pending/20260113__abcd/input/input.mp4
        # -> jobs/output/20260113__abcd.mp4
        if video_key.startswith(PENDING_PREFIX):
            # ตัด prefix 'jobs/pending/' ออก แล้วไปต่อท้ายที่ OUTPUT_PREFIX
            tail = video_key[len(PENDING_PREFIX) :]
            # ลบโฟลเดอร์ input/ ช่วงท้ายถ้ามี
            # เช่น '20260113__abcd/input/input.mp4' -> '20260113__abcd.mp4'
            if "/input/" in tail:
                folder, filename = tail.split("/input/", 1)
                tail = f"{folder}.mp4"
            result_video_key = f"{OUTPUT_PREFIX}{tail}"
        else:
            # fallback แบบง่าย ๆ
            result_video_key = f"{OUTPUT_PREFIX}{job_id}.mp4"

        try:
            copy_video(video_key, result_video_key)
        except Exception as e:
            logging.exception(
                "Failed to copy video for job %s (src=%s, dst=%s): %s",
                job_id,
                video_key,
                result_video_key,
                e,
            )
            # ถ้าก๊อปวิดีโอไม่ได้ เราถือว่า job fail เลย
            raise

    # ถ้ามาถึงตรงนี้ แปลว่าอย่างน้อยส่วนที่จำเป็นสำเร็จ
    result = {
        "status": "done",
        "job_id": job_id,
        "video_key": video_key,
        "result_video_key": result_video_key,
    }

    # เขียนผลลัพธ์ไปที่ jobs/output/{job_id}.json
    output_json_key = f"{OUTPUT_PREFIX}{job_id}.json"
    save_json(result, output_json_key)

    # ย้าย job เดิมจาก pending -> done
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

        # เสร็จหนึ่งรอบแล้ววน loop ต่อเลย


if __name__ == "__main__":
    main()
