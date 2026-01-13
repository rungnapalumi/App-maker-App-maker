# src/worker.py

import os
import time
import json
import tempfile
import logging

import boto3
from botocore.exceptions import ClientError
from moviepy.editor import VideoFileClip

from config import (
    PENDING_PREFIX,
    PROCESSING_PREFIX,
    OUTPUT_PREFIX,
    FAILED_PREFIX,
)

# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------

AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# << สำคัญที่สุด >>
# ใช้ AWS_BUCKET ก่อน ถ้าไม่มีก็ fallback ไปใช้ S3_BUCKET
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
if not AWS_BUCKET:
    raise RuntimeError(
        "Neither AWS_BUCKET nor S3_BUCKET is set in environment variables"
    )

JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "5"))

# ---------------------------------------------------------------------
# S3 client
# ---------------------------------------------------------------------
s3 = boto3.client("s3", region_name=AWS_REGION)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def list_pending_jobs():
    """List all pending job JSON objects under PENDING_PREFIX."""
    try:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    except ClientError:
        logger.exception("Error listing pending jobs from S3")
        return []

    contents = resp.get("Contents") or []
    # sort oldest first
    contents.sort(key=lambda c: c["LastModified"])
    return contents


def claim_job():
    """
    Move one job JSON from jobs/pending/ -> jobs/processing/
    and return (job_id, processing_key) or None if no job.
    """
    objects = list_pending_jobs()
    if not objects:
        return None

    obj = objects[0]
    src_key = obj["Key"]  # e.g. jobs/pending/20260112_151553__109d9b.json
    job_filename = os.path.basename(src_key)
    job_id = os.path.splitext(job_filename)[0]

    dest_key = src_key.replace(PENDING_PREFIX, PROCESSING_PREFIX, 1)

    logger.info(
        "Claim job %s: %s -> %s",
        job_id,
        src_key,
        dest_key,
    )

    copy_source = {"Bucket": AWS_BUCKET, "Key": src_key}
    s3.copy_object(Bucket=AWS_BUCKET, CopySource=copy_source, Key=dest_key)
    s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)

    return job_id, dest_key


def load_job(job_key: str) -> dict:
    """Download job JSON from S3 and parse it."""
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=job_key)
    data = obj["Body"].read()
    return json.loads(data)


def process_video_to_dots(input_path: str, output_path: str):
    """
    แปลงวิดีโอด้วย moviepy
    ตอนนี้เป็น placeholder: re-encode เฉย ๆ
    (ภายหลังค่อยใส่โค้ดสร้าง Johansson dots แทนได้)
    """
    logger.info("Opening input video with moviepy: %s", input_path)
    with VideoFileClip(input_path) as clip:
        # เขียนไฟล์ใหม่ (จะช่วยให้แน่ใจว่าเป็น mp4/codec ที่ดี)
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )
    logger.info("Wrote processed video to %s", output_path)


def process_job(job_id: str, job_key: str):
    """
    Process a single job currently in jobs/processing/.
    Expected job JSON keys:
      - video_key: S3 key ของ input video
      - mode: "dots" (optional, default "dots")
    """
    job = load_job(job_key)
    video_key = (
        job.get("video_key")
        or job.get("input_key")
        or job.get("key")
    )
    mode = job.get("mode", "dots")

    if not video_key:
        raise ValueError(f"Job {job_id} missing 'video_key' (job json: {job})")

    logger.info(
        "Processing job %s (mode=%s, video_key=%s)",
        job_id,
        mode,
        video_key,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "dots.mp4")

        # 1) Download input video
        logger.info("Downloading input video from s3://%s/%s", AWS_BUCKET, video_key)
        s3.download_file(AWS_BUCKET, video_key, input_path)

        # 2) Process video (placeholder)
        process_video_to_dots(input_path, output_path)

        # 3) Upload output video
        output_prefix = f"{OUTPUT_PREFIX}{job_id}/"
        output_key = f"{output_prefix}dots.mp4"

        logger.info(
            "Uploading output video to s3://%s/%s",
            AWS_BUCKET,
            output_key,
        )
        s3.upload_file(
            output_path,
            AWS_BUCKET,
            output_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

    # 4) Move job JSON from processing -> output (keep a record)
    done_key = job_key.replace(PROCESSING_PREFIX, OUTPUT_PREFIX, 1)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": job_key},
        Key=done_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=job_key)

    logger.info("Job %s completed successfully ✅", job_id)


def mark_job_failed(job_id: str, job_key: str, error: Exception):
    """Move job JSON from processing -> failed/ and log the error."""
    logger.exception("Job %s failed: %s", job_id, error)

    failed_key = job_key.replace(PROCESSING_PREFIX, FAILED_PREFIX, 1)
    try:
        s3.copy_object(
            Bucket=AWS_BUCKET,
            CopySource={"Bucket": AWS_BUCKET, "Key": job_key},
            Key=failed_key,
        )
        s3.delete_object(Bucket=AWS_BUCKET, Key=job_key)
    except ClientError:
        logger.exception(
            "Error while moving failed job %s from %s to %s",
            job_id,
            job_key,
            failed_key,
        )


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    logger.info(
        "Worker starting. bucket=%s region=%s poll_interval=%ss",
        AWS_BUCKET,
        AWS_REGION,
        JOB_POLL_INTERVAL,
    )

    while True:
        try:
            claimed = claim_job()
            if not claimed:
                logger.info(
                    "No pending jobs. Sleeping %s seconds...",
                    JOB_POLL_INTERVAL,
                )
                time.sleep(JOB_POLL_INTERVAL)
                continue

            job_id, job_key = claimed

            try:
                process_job(job_id, job_key)
            except Exception as e:
                mark_job_failed(job_id, job_key, e)

        except Exception:
            # ถ้า loop หลักพัง อย่าให้ container ตายทันที ให้ sleep แล้วลองใหม่
            logger.exception(
                "Unexpected error in worker loop. Sleeping %s seconds before retry.",
                JOB_POLL_INTERVAL,
            )
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main()
