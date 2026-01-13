# src/worker.py
# Background worker for AI People Reader - Dot Motion Jobs

import os
import json
import time
import tempfile
import logging

import boto3
import cv2
import numpy as np
import mediapipe as mp

# ------------------------------------------------------------------------------
# Config & Logging
# ------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("AWS_BUCKET or S3_BUCKET environment variable must be set")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
OUTPUT_PREFIX = "jobs/output/"
FAILED_PREFIX = "jobs/failed/"
DONE_PREFIX = "jobs/done/"

# mediapipe import แบบที่ถูกต้อง
mp_pose = mp.solutions.pose

# ------------------------------------------------------------------------------
# S3 Helpers
# ------------------------------------------------------------------------------

def list_pending_jobs():
    """Return first pending job json object (key + parsed json)."""
    resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    contents = resp.get("Contents", [])
    # หางานตัวแรกที่เป็น .json (ไฟล์ job definition)
    for obj in contents:
        key = obj["Key"]
        if key.endswith(".json"):
            logging.info("Found pending job json: %s", key)
            body = s3.get_object(Bucket=AWS_BUCKET, Key=key)["Body"].read()
            job = json.loads(body.decode("utf-8"))
            job["job_json_key"] = key
            return job
    return None


def move_s3_object(src_key: str, dst_key: str):
    """Copy then delete = move object in S3."""
    if src_key == dst_key:
        return
    logging.info("Moving S3 object %s -> %s", src_key, dst_key)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": src_key},
        Key=dst_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)


def upload_json(key: str, data: dict):
    logging.info("Uploading JSON to %s", key)
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(data).encode("utf-8"),
        ContentType="application/json",
    )


# ------------------------------------------------------------------------------
# Video processing (Dot Motion)
# ------------------------------------------------------------------------------

def generate_dot_video(input_path: str, output_path: str, dot_radius: int = 2):
    """
    Read a video, run Mediapipe Pose, and write new video with white dots
    at landmark positions on black background.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    logging.info(
        "Processing video for dot motion: %dx%d @ %.2f fps", width, height, fps
    )

    frame_idx = 0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            # black background
            output = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            output,
                            (cx, cy),
                            radius=dot_radius,
                            color=(255, 255, 255),
                            thickness=-1,
                        )

            out.write(output)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logging.info("Processed %d frames...", frame_idx)

    cap.release()
    out.release()
    logging.info("Dot motion video generated: %s", output_path)


# ------------------------------------------------------------------------------
# Job processing
# ------------------------------------------------------------------------------

def process_job(job: dict):
    """
    Process a single job:
    - Move job json from pending -> processing
    - Download input video
    - Generate dot video
    - Upload result video + output json
    - Write done / failed json
    """
    job_id = job.get("job_id")
    mode = job.get("mode", "dots")
    video_key = job.get("video_key")
    job_json_key = job.get("job_json_key")

    if not job_id or not video_key:
        raise ValueError(f"Invalid job payload: {job}")

    logging.info("=== Starting job %s (mode=%s) ===", job_id, mode)

    # Move job json -> processing
    processing_job_key = job_json_key.replace(PENDING_PREFIX, PROCESSING_PREFIX, 1)
    move_s3_object(job_json_key, processing_job_key)

    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_input.close()
    tmp_output.close()

    try:
        # Download input video
        logging.info("Downloading input video from s3://%s/%s", AWS_BUCKET, video_key)
        s3.download_file(AWS_BUCKET, video_key, tmp_input.name)

        # Only mode right now is "dots"
        if mode == "dots":
            generate_dot_video(tmp_input.name, tmp_output.name, dot_radius=2)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Upload output video
        output_video_key = f"{OUTPUT_PREFIX}{job_id}.mp4"
        logging.info(
            "Uploading processed video to s3://%s/%s", AWS_BUCKET, output_video_key
        )
        s3.upload_file(
            tmp_output.name,
            AWS_BUCKET,
            output_video_key,
            ExtraArgs={"ContentType": "video/mp4"},
        )

        # Write output JSON used by front-end "เช็คผลลัพธ์จาก S3"
        output_json_key = f"{OUTPUT_PREFIX}{job_id}.json"
        output_payload = {
            "status": "done",
            "job_id": job_id,
            "mode": mode,
            "video_key": output_video_key,
        }
        upload_json(output_json_key, output_payload)

        # Optional: mark job json as done
        done_job_key = f"{DONE_PREFIX}{job_id}.json"
        upload_json(done_job_key, output_payload)

        logging.info("=== Job %s completed successfully ===", job_id)

    except Exception as e:
        logging.exception("Job %s failed: %s", job_id, e)
        failed_json_key = f"{FAILED_PREFIX}{job_id}.json"
        failed_payload = {
            "status": "failed",
            "job_id": job_id,
            "mode": mode,
            "video_key": video_key,
            "error": str(e),
        }
        upload_json(failed_json_key, failed_payload)
        raise
    finally:
        try:
            os.unlink(tmp_input.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_output.name)
        except OSError:
            pass


# ------------------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------------------

def main():
    logging.info(
        "Worker starting. bucket=%s region=%s poll_interval=%ss",
        AWS_BUCKET,
        AWS_REGION,
        POLL_INTERVAL,
    )

    while True:
        try:
            job = list_pending_jobs()
            if not job:
                logging.info("No pending jobs. Sleeping %s seconds...", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
                continue

            process_job(job)

        except Exception as loop_err:
            logging.exception("Error in worker loop: %s", loop_err)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
