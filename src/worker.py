import os
import time
import traceback
from datetime import datetime, timezone

import boto3
from botocore.config import Config

AWS_REGION = (os.getenv("AWS_REGION") or "").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()

def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def get_s3():
    # kill proxy usage
    cfg = Config(proxies={}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def main():
    log("✅ Worker boot (S3 debug)")
    log(f"AWS_REGION repr: {AWS_REGION!r}")
    log(f"S3_BUCKET  repr: {S3_BUCKET!r}")

    if not AWS_REGION:
        raise RuntimeError("AWS_REGION missing/empty")
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET missing/empty")

    s3 = get_s3()
    log(f"S3 endpoint url (from boto3): {s3.meta.endpoint_url!r}")

    s3.head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        log("⏳ heartbeat")
        time.sleep(15)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("💥 crash")
        traceback.print_exc()
        raise
