import os
import json
import time
import tempfile
import boto3
from datetime import datetime

BUCKET = os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

PENDING_PREFIX = "jobs/pending/"
RUNNING_PREFIX = "jobs/running/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

POLL_SECONDS = int(os.getenv("WORKER_POLL_SECONDS", "5"))

def s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def s3_get_json(key: str) -> dict:
    client = s3()
    obj = client.get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def s3_put_json(key: str, data: dict):
    client = s3()
    client.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

def s3_exists(key: str) -> bool:
    client = s3()
    try:
        client.head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False

def list_pending_jobs(max_keys=10):
    client = s3()
    resp = client.list_objects_v2(Bucket=BUCKET, Prefix=PENDING_PREFIX, MaxKeys=max_keys)
    contents = resp.get("Contents", [])
    # oldest first
    contents.sort(key=lambda x: x.get("LastModified"))
    return [c["Key"] for c in contents]

def claim_job(pending_key: str) -> str | None:
    """
    Best-effort claim:
    - copy pending -> running (same filename)
    - delete pending
    Returns running_key if claimed else None
    """
    client = s3()
    filename = pending_key.split("/")[-1]
    running_key = f"{RUNNING_PREFIX}{filename}"

    # If already running, skip
    if s3_exists(running_key):
        return None

    try:
        client.copy_object(
            Bucket=BUCKET,
            Key=running_key,
            CopySource={"Bucket": BUCKET, "Key": pending_key},
        )
        client.delete_object(Bucket=BUCKET, Key=pending_key)
        return running_key
    except Exception as e:
        print("❌ claim_job failed:", repr(e))
        return None

def update_status(job_id: str, status: str, progress: int, message: str = "", outputs: dict | None = None):
    key = f"jobs/{job_id}/status.json"
    payload = {
        "job_id": job_id,
        "status": status,               # queued|running|done|failed
        "progress": int(progress),      # 0-100
        "message": message,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": outputs or {},
    }
    s3_put_json(key, payload)

def process_video_stub(input_path: str, output_overlay_path: str):
    """
    TODO: replace with your MediaPipe pipeline.
    For now: just copy input -> output as placeholder.
    """
    import shutil
    shutil.copyfile(input_path, output_overlay_path)

def handle_job(job: dict):
    job_id = job["job_id"]
    input_key = job["input_key"]  # e.g. jobs/{job_id}/input/input.mp4

    update_status(job_id, "running", 5, "Downloading input...")

    client = s3()
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "overlay.mp4")

        client.download_file(BUCKET, input_key, in_path)
        update_status(job_id, "running", 30, "Processing video...")

        # ====== Replace this stub with your real pipeline ======
        process_video_stub(in_path, out_path)
        # =======================================================

        update_status(job_id, "running", 80, "Uploading outputs...")
        overlay_key = f"jobs/{job_id}/output/overlay.mp4"
        client.upload_file(out_path, BUCKET, overlay_key)

    update_status(
        job_id,
        "done",
        100,
        "Completed",
        outputs={"overlay_key": overlay_key}
    )

def main():
    print("✅ Worker started (S3 queue mode)")
    print("S3_BUCKET =", BUCKET)
    print("AWS_REGION =", AWS_REGION)

    # quick check
    s3().head_bucket(Bucket=BUCKET)
    print("✅ S3 reachable")

    while True:
        try:
            pending = list_pending_jobs(max_keys=5)
            if not pending:
                time.sleep(POLL_SECONDS)
                continue

            pending_key = pending[0]
            running_key = claim_job(pending_key)
            if not running_key:
                time.sleep(1)
                continue

            job = s3_get_json(running_key)
            job_id = job["job_id"]

            try:
                handle_job(job)
                # mark done file (optional)
                s3().copy_object(
                    Bucket=BUCKET,
                    Key=f"{DONE_PREFIX}{job_id}.json",
                    CopySource={"Bucket": BUCKET, "Key": running_key},
                )
                s3().delete_object(Bucket=BUCKET, Key=running_key)
            except Exception as e:
                print("❌ Job failed:", repr(e))
                update_status(job_id, "failed", 100, message=repr(e))
                s3().copy_object(
                    Bucket=BUCKET,
                    Key=f"{FAILED_PREFIX}{job_id}.json",
                    CopySource={"Bucket": BUCKET, "Key": running_key},
                )
                s3().delete_object(Bucket=BUCKET, Key=running_key)

        except Exception as e:
            print("❌ Worker loop error:", repr(e))
            time.sleep(3)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

