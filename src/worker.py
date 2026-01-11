import os
import json
import time
import tempfile
import boto3
from datetime import datetime

# =========================
# ENV
# =========================
AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
BUCKET = (os.getenv("S3_BUCKET") or "").strip()

PENDING_PREFIX = "jobs/pending/"
RUNNING_PREFIX = "jobs/running/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())


# =========================
# S3 client (NO endpoint_url!)
# =========================
def s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )


def _require_env():
    missing = []
    if not (os.getenv("AWS_ACCESS_KEY_ID") or "").strip():
        missing.append("AWS_ACCESS_KEY_ID")
    if not (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip():
        missing.append("AWS_SECRET_ACCESS_KEY")
    if not (os.getenv("AWS_REGION") or "").strip():
        missing.append("AWS_REGION")
    if not (os.getenv("S3_BUCKET") or "").strip():
        missing.append("S3_BUCKET")
    if missing:
        raise RuntimeError("Missing env: " + ", ".join(missing))


# =========================
# JSON helpers
# =========================
def s3_get_json(key: str) -> dict:
    obj = s3().get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(key: str, data: dict):
    s3().put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def s3_exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False


def list_pending_jobs(max_keys=10):
    resp = s3().list_objects_v2(Bucket=BUCKET, Prefix=PENDING_PREFIX, MaxKeys=max_keys)
    contents = resp.get("Contents", [])
    contents.sort(key=lambda x: x.get("LastModified"))
    return [c["Key"] for c in contents]


def claim_job(pending_key: str) -> str | None:
    """
    Best-effort claim:
      - copy pending -> running
      - delete pending
    """
    filename = pending_key.split("/")[-1]
    running_key = f"{RUNNING_PREFIX}{filename}"

    if s3_exists(running_key):
        return None

    try:
        s3().copy_object(
            Bucket=BUCKET,
            Key=running_key,
            CopySource={"Bucket": BUCKET, "Key": pending_key},
        )
        s3().delete_object(Bucket=BUCKET, Key=pending_key)
        return running_key
    except Exception as e:
        print("❌ claim_job failed:", repr(e))
        return None


def update_status(job_id: str, status: str, progress: int, message: str = "", outputs: dict | None = None):
    key = f"jobs/{job_id}/status.json"
    payload = {
        "job_id": job_id,
        "status": status,          # queued|running|done|failed
        "progress": int(progress), # 0-100
        "message": message,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": outputs or {},
    }
    s3_put_json(key, payload)


# =========================
# Processing (stub)
# =========================
def process_video_stub(input_path: str, overlay_out_path: str):
    """
    TODO: Replace with your MediaPipe pipeline later.
    Now: copy input -> output to prove flow works.
    """
    import shutil
    shutil.copyfile(input_path, overlay_out_path)


def handle_job(job: dict):
    job_id = job["job_id"]
    input_key = job["input_key"]

    update_status(job_id, "running", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "overlay.mp4")

        s3().download_file(BUCKET, input_key, in_path)
        update_status(job_id, "running", 30, "Processing video...")

        process_video_stub(in_path, out_path)

        update_status(job_id, "running", 80, "Uploading outputs...")
        overlay_key = f"jobs/{job_id}/output/overlay.mp4"
        s3().upload_file(out_path, BUCKET, overlay_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": overlay_key})


# =========================
# Main loop
# =========================
def main():
    print("✅ Worker started (S3 queue mode)")
    print("AWS_REGION =", AWS_REGION)
    print("S3_BUCKET  =", BUCKET)

    _require_env()

    # quick S3 check
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
            job_id = job.get("job_id", "unknown")

            try:
                print("▶️ Processing job:", job_id)
                handle_job(job)

                # mark done ticket (optional)
                s3().copy_object(
                    Bucket=BUCKET,
                    Key=f"{DONE_PREFIX}{job_id}.json",
                    CopySource={"Bucket": BUCKET, "Key": running_key},
                )
                s3().delete_object(Bucket=BUCKET, Key=running_key)
                print("✅ Job done:", job_id)

            except Exception as e:
                print("❌ Job failed:", job_id, repr(e))
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

def s3_get_json(key: str) -> dict:
    obj = s3().get_object(Bucket=BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def s3_put_json(key: str, data: dict):
    s3().put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

def s3_exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False

def list_pending_jobs(max_keys=10):
    resp = s3().list_objects_v2(Bucket=BUCKET, Prefix=PENDING_PREFIX, MaxKeys=max_keys)
    contents = resp.get("Contents", [])
    contents.sort(key=lambda x: x.get("LastModified"))
    return [c["Key"] for c in contents]

def claim_job(pending_key: str) -> str | None:
    """
    Best-effort claim:
      - copy pending -> running
      - delete pending
    """
    filename = pending_key.split("/")[-1]
    running_key = f"{RUNNING_PREFIX}{filename}"

    if s3_exists(running_key):
        return None

    try:
        s3().copy_object(
            Bucket=BUCKET,
            Key=running_key,
            CopySource={"Bucket": BUCKET, "Key": pending_key},
        )
        s3().delete_object(Bucket=BUCKET, Key=pending_key)
        return running_key
    except Exception as e:
        print("❌ claim_job failed:", repr(e))
        return None

def update_status(job_id: str, status: str, progress: int, message: str = "", outputs: dict | None = None):
    key = f"jobs/{job_id}/status.json"
    payload = {
        "job_id": job_id,
        "status": status,          # queued|running|done|failed
        "progress": int(progress), # 0-100
        "message": message,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": outputs or {},
    }
    s3_put_json(key, payload)

# ====================
# Processing (stub -> replace later)
# ====================
def process_video_stub(input_path: str, overlay_out_path: str):
    """
    TODO: Replace with your MediaPipe pipeline.
    For now: copy input -> overlay output (proves full pipeline works).
    """
    import shutil
    shutil.copyfile(input_path, overlay_out_path)

def handle_job(job: dict):
    job_id = job["job_id"]
    input_key = job["input_key"]

    update_status(job_id, "running", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "overlay.mp4")

        s3().download_file(BUCKET, input_key, in_path)
        update_status(job_id, "running", 30, "Processing video...")

        # ===== replace stub with real pipeline later =====
        process_video_stub(in_path, out_path)
        # ===============================================

        update_status(job_id, "running", 80, "Uploading outputs...")
        overlay_key = f"jobs/{job_id}/output/overlay.mp4"
        s3().upload_file(out_path, BUCKET, overlay_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": overlay_key})

# ====================
# Main loop
# ====================
def main():
    print("✅ Worker started (S3 queue mode)")
    print("AWS_REGION =", AWS_REGION)
    print("S3_BUCKET  =", BUCKET)

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

                # mark done ticket
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
