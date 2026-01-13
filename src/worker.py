import os
import json
import boto3
import tempfile
import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import VideoFileClip

# Mediapipe objects
mp_pose = mp.solutions.pose

# S3 settings
BUCKET = "ai-people-reader-storage"

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
OUTPUT_PREFIX = "jobs/output/"
FAILED_PREFIX = "jobs/failed/"

s3 = boto3.client("s3")


def claim_job():
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=PENDING_PREFIX)
    if "Contents" not in objects:
        return None

    for obj in objects["Contents"]:
        key = obj["Key"]
        if key.endswith(".json"):
            dest = PROCESSING_PREFIX + os.path.basename(key)
            s3.copy_object(
                Bucket=BUCKET,
                CopySource=f"{BUCKET}/{key}",
                Key=dest
            )
            s3.delete_object(Bucket=BUCKET, Key=key)
            return dest

    return None


def download_input(processing_key):
    job_json = s3.get_object(Bucket=BUCKET, Key=processing_key)
    job_data = json.loads(job_json["Body"].read())

    video_key = job_data["video_key"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        s3.download_file(BUCKET, video_key, tmp.name)
        return tmp.name, job_data


def create_johansson_dots(input_video_path):
    pose = mp_pose.Pose(static_image_mode=False)

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        dots_frame = np.zeros_like(frame)

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * width)
                y = int(lm.y * height)
                cv2.circle(dots_frame, (x, y), 5, (255, 255, 255), -1)

        writer.write(dots_frame)

    cap.release()
    writer.release()
    pose.close()

    return output_path


def upload_output(job_id, video_path):
    output_key = f"{OUTPUT_PREFIX}{job_id}/dots.mp4"
    s3.upload_file(video_path, BUCKET, output_key)
    return output_key


def mark_failed(processing_key, reason):
    failed_key = FAILED_PREFIX + os.path.basename(processing_key)
    s3.copy_object(
        Bucket=BUCKET,
        CopySource=f"{BUCKET}/{processing_key}",
        Key=failed_key
    )
    s3.delete_object(Bucket=BUCKET, Key=processing_key)

    print("FAILED:", reason)


def handle_job(processing_key):
    try:
        input_video_path, job_data = download_input(processing_key)
        job_id = job_data["job_id"]

        print(f"Creating Johansson dots for job {job_id}...")
        output_path = create_johansson_dots(input_video_path)

        output_key = upload_output(job_id, output_path)
        print("Uploaded output ->", output_key)

        s3.delete_object(Bucket=BUCKET, Key=processing_key)
    except Exception as e:
        mark_failed(processing_key, str(e))


def main():
    print("Worker started... waiting for jobs.")
    while True:
        job_key = claim_job()
        if job_key:
            print("Claimed job:", job_key)
            handle_job(job_key)
        else:
            import time
            time.sleep(5)


if __name__ == "__main__":
    main()
