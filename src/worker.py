# src/worker.py

import os
import json
import time
import logging
import tempfile

import boto3
from botocore.exceptions import ClientError
import cv2
import numpy as np
import mediapipe as mp

from config import PENDING_PREFIX, PROCESSING_PREFIX, OUTPUT_PREFIX, FAILED_PREFIX

# ----------------- CONFIG / CLIENT -----------------

AWS_BUCKET = os.environ["AWS_BUCKET"]          # ตั้งใน Render dashboard
REGION_NAME = os.environ.get("AWS_REGION", "ap-southeast-1")

s3 = boto3.client("s3", region_name=REGION_NAME)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("worker")

mp_pose = mp.solutions.pose


# ----------------- S3 HELPERS -----------------


def list_pending_job_keys():
    """คืน list ของ key jobs/pending/*.json"""
    resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    contents = resp.get("Contents") or []
    return [obj["Key"] for obj in contents if obj["Key"].endswith(".json")]


def claim_job(job_key: str) -> dict:
    """
    ย้ายไฟล์ job จาก pending -> processing แล้วอ่านเนื้อหา
    """
    job_id = os.path.basename(job_key).replace(".json", "")
    processing_key = f"{PROCESSING_PREFIX}{job_id}.json"

    logger.info("Claim job %s -> %s", job_key, processing_key)

    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": job_key},
        Key=processing_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=job_key)

    obj = s3.get_object(Bucket=AWS_BUCKET, Key=processing_key)
    job = json.loads(obj["Body"].read())

    job["id"] = job_id
    job["processing_key"] = processing_key
    return job


def download_to_temp(s3_key: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    logger.info("Downloading %s -> %s", s3_key, path)
    s3.download_file(AWS_BUCKET, s3_key, path)
    return path


def upload_from_path(path: str, s3_key: str, content_type: str):
    logger.info("Uploading %s -> %s", path, s3_key)
    s3.upload_file(
        path,
        AWS_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )


def write_result(job: dict, output_s3_key: str):
    job_id = job["id"]
    result_key = f"{OUTPUT_PREFIX}{job_id}.json"
    payload = {
        "status": "done",
        "job_id": job_id,
        "mode": job.get("mode", "dots"),
        "video_key": output_s3_key,
    }
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=result_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Job %s finished -> %s (meta: %s)", job_id, output_s3_key, result_key)


def mark_failed(job: dict, message: str):
    job_id = job.get("id", "unknown")
    failed_key = f"{FAILED_PREFIX}{job_id}.json"
    payload = {
        "status": "failed",
        "job_id": job_id,
        "error": message,
    }
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=failed_key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    logger.error("Job %s FAILED: %s", job_id, message)


# ----------------- VIDEO PROCESSING -----------------


def make_johansson_dots(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            blank = np.zeros_like(frame)

            if results.pose_landmarks:
                hh, ww, _ = blank.shape
                for lm in results.pose_landmarks.landmark:
                    x = int(lm.x * ww)
                    y = int(lm.y * hh)
                    cv2.circle(blank, (x, y), 4, (255, 255, 255), -1)

            out.write(blank)

    cap.release()
    out.release()


def make_skeleton_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )

            out.write(frame)

    cap.release()
    out.release()


def process_job(job: dict) -> str:
    """
    job json ควรมีอย่างน้อย:
        {
          "video_key": "jobs/2026.../input/input.mp4",
          "mode": "dots" หรือ "skeleton"
        }
    """
    mode = job.get("mode", "dots")
    video_key = job["video_key"]
    job_id = job["id"]

    logger.info("Processing job %s: video_key=%s, mode=%s", job_id, video_key, mode)

    input_path = download_to_temp(video_key, ".mp4")
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    if mode == "skeleton":
        make_skeleton_video(input_path, output_path)
        out_name = "skeleton.mp4"
    else:
        make_johansson_dots(input_path, output_path)
        out_name = "dots.mp4"

    output_key = f"{OUTPUT_PREFIX}{job_id}/{out_name}"
    upload_from_path(output_path, output_key, "video/mp4")
    return output_key


# ----------------- MAIN LOOP -----------------


def main():
    logger.info("Worker starting. Bucket=%s, region=%s", AWS_BUCKET, REGION_NAME)

    while True:
        try:
            pending = list_pending_job_keys()
            if not pending:
                logger.info("No pending jobs. Sleeping 5s...")
                time.sleep(5)
                continue

            # เลือก job แรกสุด
            job_key = sorted(pending)[0]
            job = claim_job(job_key)

            try:
                output_key = process_job(job)
                write_result(job, output_key)
            except Exception as e:
                logger.exception("Job %s failed", job.get("id"))
                mark_failed(job, str(e))

        except ClientError:
            logger.exception("AWS error, sleep 10s")
            time.sleep(10)
        except Exception:
            logger.exception("Unexpected error, sleep 10s")
            time.sleep(10)


if __name__ == "__main__":
    main()
