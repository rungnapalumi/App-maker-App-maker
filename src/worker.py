import os
import time
import boto3

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "ap-southeast-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

def main():
    bucket = os.getenv("S3_BUCKET")
    print("✅ worker started")
    print("AWS_REGION =", os.getenv("AWS_REGION"))
    print("S3_BUCKET  =", bucket)

    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=bucket)
        print("✅ S3 reachable")
    except Exception as e:
        print("❌ S3 error:", repr(e))

    while True:
        print("⏳ heartbeat")
        time.sleep(30)

if __name__ == "__main__":
    main()

