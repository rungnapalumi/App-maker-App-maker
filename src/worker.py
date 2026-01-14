# worker.py — AI People Reader Worker (compat version)
#
# รองรับได้ทั้ง 2 schema:
#   v1: video_key, result_video_key
#   v2: input_key, output_key

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# Config / S3 client
# ----------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
JOB_POLL_INTERVAL = float(os.environ.get("JOB_POLL_INTERVAL", "10"))  # seconds

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET / S3_BUCKET environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"

print("====== AI People Reader Worker ======", flush=True)
print(f"Using bucket: {AWS_BUCKET}", flush=True)
print(f"Region     : {AWS_REGION}", flush=True)
print(f"Poll every : {JOB_POLL_INTERVAL} seconds", flush=True)


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> Dict[str, Any]:
    print(f"[s3_get_json] key={key}", flush=True)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    print(f"[s3_put_json] key={key} size={len(body)} bytes", flush=True)
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def copy_object(src_key: str, dst_key: str) -> None:
    """
    Copy object ใน bucket เดียวกันตาม spec ของ S3
    """
    print(f"[copy_object] {src_key} -> {dst_key}", flush=True)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        Key=dst_key,
        CopySource={
            "Bucket": AWS_BUCKET,
            "Key": src_key,
        },
    )


def find_one_pending_job_key() -> Optional[str]:
    """
    หา job .json ตัวแรกใน jobs/pending/
    """
    print(f"[find_one_pending_job_key] prefix={JOBS_PENDING_PREFIX}", flush=True)
    resp = s3.list_objects_v2(
        Bucket=AWS_BUCKET,
        Prefix=JOBS_PENDING_PREFIX,
        MaxKeys=1,
    )
    contents = resp.get("Contents")
    if not contents:
        print("[find_one_pending_job_key] no pending jobs", flush=True)
        return None

    key = contents[0]["Key"]
    print(f"[find_one_pending_job_key] found {key}", flush=True)
    return key


# ----------------------------------------------------------
# Core job processing
# ----------------------------------------------------------

def process_job(pending_key: str) -> None:
    """
    ประมวลผล job หนึ่งตัวจาก jobs/pending/<job_id>.json
    """

    print(f"[process_job] start pending_key={pending_key}", flush=True)

    job = s3_get_json(pending_key)

    job_id = job.get("job_id")
    if not job_id:
        job_id = os.path.splitext(os.path.basename(pending_key))[0]
        job["job_id"] = job_id

    # รองรับได้ทั้ง 2 schema:
    # v2 (ใหม่): input_key, output_key
    # v1 (เก่า): video_key, result_video_key
    input_key = job.get("input_key") or job.get("video_key")
    output_key = job.get("output_key") or job.get("result_video_key")

    if not input_key:
        raise RuntimeError("Job JSON missing 'input_key' / 'video_key'")
    if not output_key:
        raise RuntimeError("Job JSON missing 'output_key' / 'result_video_key'")

    processing_key = pending_key.replace(JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX)

    job["status"] = "processing"
    job["updated_at"] = utc_now_iso()
    job.setdefault("error", None)

    s3_put_json(processing_key, job)
    s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)
    print("[process_job] moved JSON pending -> processing", flush=True)

    try:
        # งานจริง: copy วิดีโอ input -> output
        print(f"[process_job] copying video {input_key} -> {output_key}", flush=True)
        copy_object(input_key, output_key)

        finished_key = processing_key.replace(
            JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX
        )
        now = utc_now_iso()
        job["status"] = "finished"
        job["finished_at"] = now
        job["updated_at"] = now
        job["error"] = None

        s3_put_json(finished_key, job)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)
        print("[process_job] moved JSON processing -> finished", flush=True)

    except Exception as exc:
        print(f"[process_job] ERROR: {exc}", flush=True)

        failed_key = processing_key.replace(
            JOBS_PROCESSING_PREFIX, JOBS_FAILED_PREFIX
        )
        now = utc_now_iso()
        job["status"] = "failed"
        job["failed_at"] = now
        job["updated_at"] = now
        job["error"] = str(exc)

        s3_put_json(failed_key, job)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)
        print("[process_job] moved JSON processing -> failed", flush=True)

        raise


# ----------------------------------------------------------
# Main loop
# ----------------------------------------------------------

def main() -> None:
    print("[main] worker started", flush=True)
    while True:
        try:
            job_key = find_one_pending_job_key()
            if not job_key:
                time.sleep(JOB_POLL_INTERVAL)
                continue

            process_job(job_key)

        except ClientError as ce:
            print(f"[main] AWS ClientError: {ce}", flush=True)
            time.sleep(JOB_POLL_INTERVAL)

        except Exception as e:
            print(f"[main] Unexpected error: {e}", flush=True)
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main()
