import os
import time
import json
import traceback
from datetime import datetime, timezone

import boto3
from botocore.exceptions import ClientError

# =========================
# CONFIG
# =========================
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
BUCKET = os.getenv("S3_BUCKET")

POLL_SECONDS = 5

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"

# =========================
# AWS CLIENT
# =========================
def s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

# =========================
# UTIL
# =========================
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def log(msg):
    print(msg, flush=True)

# =========================
# JOB IO
# =========================
def list_pending_jobs(s3):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PENDING_PREFIX)
    return [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".json")]

def load_job(s3, key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def save_json(s3, key, data):
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def move_job(s3, old_key, new_key):
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": old_key},
        Key=new_key,
    )
    s3.delete_object(Bucket=BUCKET, Key=old_key)

# =========================
# VIDEO PROCESS (PLACEHOLDER)
# =========================
def process_video(input_key, output_key):
    """
    TODO:
    - Replace this with REAL dot / skeleton overlay pipeline
    - ตอนนี้เป็น placeholder: copy input -> output
    """
    s3 = s3_client()
    log(f"🎬 Processing video: {input_key}")
    s3.copy_object(
        Bucket=BUCKET,
        CopySource={"Bucket": BUCKET, "Key": input_key},
        Key=output_key,
    )

# =========================
# JOB HANDLER
# =========================
def handle_job(job_key):
    s3 = s3_client()

    job_id = os.path.basename(job_key).replace(".json", "")
    log(f"▶️ Processing job: {job_id}")

    job = load_job(s3, job_key)

    input_key = job["input_key"]
    output_key = f"jobs/{job_id}/output/overlay.mp4"

    # mark processing
    job["status"] = "processing"
    job["progress"] = 10
    job["updated_at"] = now_iso()

    processing_key = f"{PROCESSING_PREFIX}{job_id}.json"
    save_json(s3, processing_key, job)
    move_job(s3, job_key, processing_key)

    try:
        # ---- DO WORK ----
        process_video(input_key, output_key)

        # ---- DONE ----
        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Completed"
        job["outputs"] = {"overlay_key": output_key}
        job["updated_at"] = now_iso()

        done_key = f"{DONE_PREFIX}{job_id}.json"
        save_json(s3, done_key, job)
        s3.delete_object(Bucket=BUCKET, Key=processing_key)

        log(f"✅ Job done: {job_id}")

    except Exception as e:
        log(f"❌ Job failed: {job_id}")
        log(traceback.format_exc())

        job["status"] = "error"
        job["message"] = str(e)
        job["updated_at"] = now_iso()
        save_json(s3, processing_key, job)

# =========================
# MAIN LOOP
# =========================
def main():
    log("✅ Worker started (S3 queue mode)")
    log(f"AWS_REGION = {AWS_REGION}")
    log(f"S3_BUCKET  = {BUCKET}")

    if not BUCKET:
        log("❌ S3_BUCKET is not set")
        return

    s3 = s3_client()

    while True:
        try:
            pending = list_pending_jobs(s3)

            if not pending:
                log("⏳ heartbeat (no pending jobs)")
                time.sleep(POLL_SECONDS)
                continue

            for job_key in pending:
                handle_job(job_key)

        except ClientError as e:
            log("❌ AWS error")
            log(str(e))
            time.sleep(POLL_SECONDS)

        except Exception:
            log("❌ Unexpected error")
            log(traceback.format_exc())
            time.sleep(POLL_SECONDS)

# =========================
if __name__ == "__main__":
    main()
