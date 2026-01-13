import os
import time
import json
import tempfile
import logging

import boto3
import cv2
import numpy as np
import mediapipe as mp

# --------------------------------------------------------
# Environment config
# --------------------------------------------------------
S3_BUCKET = os.environ.get("S3_BUCKET") or os.environ.get("AWS_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
JOB_POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL", "10"))

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET or AWS_BUCKET environment variable is required")

s3 = boto3.client("s3", region_name=AWS_REGION)

# --------------------------------------------------------
# Mediapipe Pose (รองรับทั้งเวอร์ชันเก่าและใหม่)
# --------------------------------------------------------
try:
    # เวอร์ชันใหม่ ๆ ของ mediapipe
    from mediapipe.python.solutions import pose as mp_pose
except Exception:
    # ถ้าเวอร์ชันเก่ายังมี mp.solutions.pose
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        mp_pose = mp.solutions.pose
    else:
        raise

DOT_RADIUS = 2  # ขนาดจุดในวิดีโอ (pixel)


# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------
def download_s3_to_temp(key: str, suffix: str) -> str:
    """Download S3 object to a temporary local file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    logging.info(f"Downloading s3://{S3_BUCKET}/{key} -> {path}")
    s3.download_file(S3_BUCKET, key, path)
    return path


def upload_file_to_s3(path: str, key: str, content_type: str | None = None) -> None:
    """Upload local file to S3."""
    extra = {"ContentType": content_type} if content_type else None
    logging.info(f"Uploading {path} -> s3://{S3_BUCKET}/{key}")
    if extra:
        s3.upload_file(path, S3_BUCKET, key, ExtraArgs=extra)
    else:
        s3.upload_file(path, S3_BUCKET, key)


def put_status(job_id: str, status: str, payload: dict) -> None:
    """เขียนสถานะงานลง S3 ทั้งใน output/ และ done/ หรือ failed/"""
    body = dict(payload)
    body["job_id"] = job_id
    body["status"] = status
    data = json.dumps(body).encode("utf-8")

    # ไฟล์ผลลัพธ์หลัก
    output_key = f"jobs/output/{job_id}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=output_key,
        Body=data,
        ContentType="application/json",
    )

    # marker สำหรับดูสถานะเร็ว ๆ
    marker_key = f"jobs/{status}/{job_id}.json"
    s3.put_object(
        Bucket=S3_BUCKET,
        Key=marker_key,
        Body=data,
        ContentType="application/json",
    )

    logging.info(f"Wrote status '{status}' for job {job_id}")


def get_next_pending_job() -> dict | None:
    """หา job ตัวถัดไปจาก jobs/pending/ (เลือกไฟล์ .json แรกตามลำดับชื่อ)"""
    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix="jobs/pending/",
    )
    contents = resp.get("Contents", [])
    json_keys = [c["Key"] for c in contents if c["Key"].endswith(".json")]
    if not json_keys:
        return None

    json_keys.sort()
    key = json_keys[0]
    logging.info(f"Found pending job description: {key}")

    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    job_doc = json.loads(obj["Body"].read().decode("utf-8"))
    job_id = job_doc["job_id"]

    # ย้ายไฟล์ไปไว้ใน processing/ เพื่อกันชนกัน
    processing_key = f"jobs/processing/{job_id}.json"
    s3.copy_object(
        Bucket=S3_BUCKET,
        CopySource={"Bucket": S3_BUCKET, "Key": key},
        Key=processing_key,
    )
    s3.delete_object(Bucket=S3_BUCKET, Key=key)

    logging.info(f"Moved {key} -> {processing_key}")
    return job_doc


# --------------------------------------------------------
# Core video processing (dot motion)
# --------------------------------------------------------
def process_job(job_doc: dict) -> None:
    job_id = job_doc["job_id"]
    mode = job_doc.get("mode", "dots")
    video_key = job_doc["video_key"]

    logging.info(f"Processing job_id={job_id}, mode={mode}, video_key={video_key}")

    # 1) Download input video
    input_path = download_s3_to_temp(video_key, ".mp4")

    # 2) Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 3) Run Mediapipe Pose + วาดจุด
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            black = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(
                            black,
                            (cx, cy),
                            DOT_RADIUS,
                            (255, 255, 255),
                            thickness=-1,
                        )

            out.write(black)

    cap.release()
    out.release()

    # 4) Upload processed video
    output_video_key = f"jobs/output/{job_id}.mp4"
    upload_file_to_s3(output_path, output_video_key, "video/mp4")

    # 5) Cleanup local files
    try:
        os.remove(input_path)
    except OSError:
        pass
    try:
        os.remove(output_path)
    except OSError:
        pass

    # 6) Mark job as done
    put_status(
        job_id,
        "done",
        {"output_video_key": output_video_key, "mode": mode},
    )

    logging.info(f"Job {job_id} completed successfully.")


# --------------------------------------------------------
# Main loop
# --------------------------------------------------------
def main_loop() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(
        f"Worker started. Bucket={S3_BUCKET}, region={AWS_REGION}, "
        f"poll_interval={JOB_POLL_INTERVAL}s"
    )

    while True:
        try:
            job = get_next_pending_job()
            if job is None:
                logging.info(
                    f"No pending jobs. Sleeping {JOB_POLL_INTERVAL} seconds..."
                )
                time.sleep(JOB_POLL_INTERVAL)
                continue

            job_id = job.get("job_id", "unknown")
            try:
                process_job(job)
            except Exception as e:
                logging.exception(f"Error while processing job {job_id}")
                put_status(job_id, "failed", {"error": str(e)})

        except Exception:
            # error ระดับ loop ใหญ่ – log ไว้แล้วค่อยนอนแล้ววนใหม่
            logging.exception("Unexpected error in main loop")
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
