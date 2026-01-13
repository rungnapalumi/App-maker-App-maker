import os
import json
import boto3
import time
from botocore.exceptions import ClientError

AWS_BUCKET = os.environ["AWS_BUCKET"]
AWS_REGION = os.environ["AWS_REGION"]
POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL", "10"))

s3 = boto3.client("s3", region_name=AWS_REGION)


def safe_s3_download(bucket, key, local_path):
    """
    ปรับให้รองรับทั้ง input.mp4 และ input/input.mp4
    """
    try:
        s3.head_object(Bucket=bucket, Key=key)
        s3.download_file(bucket, key, local_path)
        print(f"[OK] Downloaded: {key}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"[MISS] Not found: {key}")
            return False
        else:
            raise e


def find_input_video(job_id):
    """
    หาวิดีโอจาก S3 — รองรับทุกแบบ
    """
    base = f"jobs/pending/{job_id}"

    candidates = [
        f"{base}/input/input.mp4",
        f"{base}/input.mp4",
        f"{base}/video.mp4",
        f"{base}/input/video.mp4",
    ]

    for key in candidates:
        print(f"Trying: {key}")
        if safe_s3_download(AWS_BUCKET, key, "/tmp/input.mp4"):
            return "/tmp/input.mp4"

    return None


def process_video(input_path, output_path):
    """
    >>> ตรงนี้ Rung ใส่ model detection/dots ของตัวเอง <<<
    """
    with open(output_path, "w") as f:
        f.write(json.dumps({"status": "done", "message": "Video processed"}))


def write_output(job_id, data):
    key = f"jobs/output/{job_id}.json"
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(data),
        ContentType="application/json",
    )
    print(f"Written output → {key}")


def write_failed(job_id, message):
    key = f"jobs/failed/{job_id}.json"
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps({"job_id": job_id, "status": "failed", "error": message}),
        ContentType="application/json",
    )
    print(f"Written FAIL → {key}")


def load_job_json(job_id):
    key = f"jobs/pending/{job_id}.json"
    tmp = "/tmp/job.json"

    try:
        s3.download_file(AWS_BUCKET, key, tmp)
        with open(tmp, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error loading job json:", e)
        return None


def main():
    print("Worker started. Polling for jobs...")

    while True:
        try:
            prefix = "jobs/pending/"
            resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=prefix)

            if "Contents" not in resp:
                print("No pending jobs. Sleeping...")
                time.sleep(POLL_INTERVAL)
                continue

            pending = [
                obj["Key"]
                for obj in resp["Contents"]
                if obj["Key"].endswith(".json")
            ]

            if not pending:
                print("No pending jobs. Sleeping...")
                time.sleep(POLL_INTERVAL)
                continue

            for job_key in pending:
                job_id = job_key.split("/")[-1].replace(".json", "")
                print(f"--- Found job: {job_id} ---")

                job = load_job_json(job_id)
                if not job:
                    write_failed(job_id, "Cannot load job JSON")
                    continue

                print("Searching for input video...")
                input_path = find_input_video(job_id)

                if not input_path:
                    write_failed(job_id, "Input video missing")
                    continue

                output_tmp = "/tmp/output.json"
                process_video(input_path, output_tmp)

                write_output(job_id, {"status": "done", "job_id": job_id})

                print(f"Job {job_id} finished.")

        except Exception as e:
            print("Worker error:", e)

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
