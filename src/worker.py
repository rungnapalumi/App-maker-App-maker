import os
import time
import json
import traceback
from datetime import datetime, timezone

import boto3
from botocore.config import Config


AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()

def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def get_s3():
    # Kill any proxy overrides at runtime
    cfg = Config(
        proxies=None,
        retries={"max_attempts": 5, "mode": "standard"},
    )
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def main():
    log("✅ Worker boot")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is missing")

    s3 = get_s3()

    # connectivity check
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
