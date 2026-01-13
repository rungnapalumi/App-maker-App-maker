# src/worker.py  --- Background worker สำหรับประมวลผล DOT video
import os
import json
import time
import tempfile
import logging

import boto3
from botocore.exceptions import ClientError

import cv2
import mediapipe as mp
import numpy as np

# moviepy เป็น optional – ถ้าไม่มีจะไม่ใส่เสียง
try:
    from moviepy.editor import VideoFileClip
except Exception:  # ModuleNotFoundError หรืออย่างอื่น
    VideoFileClip = None

# ----------------------------------------------------------
# Environment
# ----------------------------------------------------------
AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
JOB_POLL_INTERVAL = int(os.environ.get("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

mp_pose = mp.solutions.pose

# ----------------------------------------------------------
# S3 helpers
# ----------------------------------------------------------
def download_to_temp(bucket: str, key: str) -> str:
    """ดาวน์โหลดไฟล์จาก S3 ลง temp แล้วคืน path กลับมา"""
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    logging.info(f"Downloading s3://{bucket}/{key} -> {temp_path}")
    s3.download_file(bucket, key, temp_path)
    return temp_path

def upload_file(bucket: str, key: str, local_path: str):
    logging.info(f"Uploading {local_path} -> s3://{bucket}/{key}")
    s3.upload_file(local_path, bucket, key)

def put_json(bucket: str, key: str, payload: dict):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.close()
    upload_file(bucket, key, tmp.name)
    os.unlink(tmp.name)

def list_pending_job_ids():
    """หาไฟล์ jobs/pending/*.json แล้วดึง job_id ออกมา"""
    prefix = "jobs/pending/"
    resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=prefix)

    job_ids = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        # ต้องเป็นไฟล์ .json ที่อยู่ตรงใต้ pending เลย (ไม่มี / เพิ่ม)
        if key.startswith(prefix) and key.endswith(".json") and key.count("/") == 2:
            job_id = key.split("/")[-1].removesuffix(".json")
            job_ids.append(job_id)
    return job_ids

def move_s3_object(old_key: str, new_key: str):
    """จำลอง move โดย copy แล้ว delete"""
    logging.info(f"Moving s3://{AWS_BUCKET}/{old_key} -> s3://{AWS_BUCKET}/{new_key}")
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": old_key},
        Key=new_key,
    )
    s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)

# ----------------------------------------------------------
# DOT generator core
# ----------------------------------------------------------
def generate_dot_motion_video(input_path: str, output_dir: str, dot_size: int = 2):
    """
    แปลงวิดีโอจริงเป็น dot-motion
    คืนค่า:
      - output_no_audio: path วิดีโอ dot ไม่มีเสียง
      - output_with_audio: path วิดีโอ dot ใส่เสียงแล้ว (หรือ None ถ้าทำไม่ได้)
    """
    os.makedirs(output_dir, exist_ok=True)

    output_no_audio = os.path.join(output_dir, "dots_no_audio.mp4")
    output_with_audio = os.path.join(output_dir, "dots_with_audio.mp4")

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_no_audio, fourcc, fps, (width, height))

    logging.info(f"DOT processing start: {input_path} -> {output_no_audio}")

    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            blank = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.circle(blank, (cx, cy), dot_size, (255, 255, 255), -1)

            writer.write(blank)

    cap.release()
    writer.release()

    # ลองใส่เสียง (ถ้ามี moviepy)
    if VideoFileClip is None:
        logging.info("moviepy ไม่พร้อมใช้งาน ข้ามขั้นตอนรวมเสียง")
        return output_no_audio, None

    try:
        logging.info("Trying to merge original audio into dot video...")
        original = VideoFileClip(input_path)
        processed = VideoFileClip(output_no_audio)

        if original.audio is not None:
            final = processed.set_audio(original.audio)
            final.write_videofile(
                output_with_audio,
                codec="libx264",
                audio_codec="aac",
                verbose=False,
                logger=None,
            )
            original.close()
            processed.close()
            final.close()
            logging.info("Audio merged successfully")
            return output_no_audio, output_with_audio
        else:
            logging.info("Original video has no audio")
    except Exception as e:
        logging.warning(f"Failed to merge audio: {e}")

    return output_no_audio, None

def process_video_dots(local_video_path: str, job_id: str, result_prefix: str):
    """
    ใช้สำหรับ worker ดึง dot video แล้ว upload กลับ S3
    """
    local_output_dir = f"/tmp/{job_id}_dots"
    os.makedirs(local_output_dir, exist_ok=True)

    no_audio_path, with_audio_path = generate_dot_motion_video(
        local_video_path,
        local_output_dir,
        dot_size=2,
    )

    result_files = []

    # upload video (no audio)
    s3_key_no_audio = f"{result_prefix}{job_id}_dots_no_audio.mp4"
    upload_file(AWS_BUCKET, s3_key_no_audio, no_audio_path)
    result_files.append(s3_key_no_audio)

    # upload video with audio if present
    if with_audio_path:
        s3_key_with_audio = f"{result_prefix}{job_id}_dots_with_audio.mp4"
        upload_file(AWS_BUCKET, s3_key_with_audio, with_audio_path)
        result_files.append(s3_key_with_audio)

    return {
        "result_prefix": result_prefix,
        "result_files": result_files,
    }

# ----------------------------------------------------------
# Job processing
# ----------------------------------------------------------
def handle_job(job_id: str):
    logging.info(f"Processing job_id={job_id}")

    pending_meta_key = f"jobs/pending/{job_id}.json"

    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=pending_meta_key)
        job_meta = json.loads(obj["Body"].read())
    except ClientError as e:
        logging.error(f"Cannot read job metadata {pending_meta_key}: {e}")
        return

    mode = job_meta.get("mode", "dots")
    video_key = job_meta["video_key"]

    try:
        # โหลดวิดีโอจาก S3
        local_video_path = download_to_temp(AWS_BUCKET, video_key)

        # prefix สำหรับผลลัพธ์
        result_prefix = f"jobs/output-data/{job_id}/"

        if mode == "dots":
            result_info = process_video_dots(local_video_path, job_id, result_prefix)
        else:
            # เผื่ออนาคตมี mode อื่น
            logging.warning(f"Unknown mode '{mode}', skip processing")
            result_info = {
                "result_prefix": result_prefix,
                "result_files": [],
                "warning": f"Unknown mode {mode}",
            }

        # เขียน summary -> jobs/output/<job_id>.json
        output_summary = {
            "job_id": job_id,
            "mode": mode,
            "status": "done",
            "video_key": video_key,
            "result": result_info,
        }
        output_key = f"jobs/output/{job_id}.json"
        put_json(AWS_BUCKET, output_key, output_summary)

        # ย้าย metadata ไปโฟลเดอร์ done และ (ถ้าต้องการ) ลบ pending
        move_s3_object(pending_meta_key, f"jobs/done/{job_id}.json")

        logging.info(f"Job {job_id} done")

    except Exception as e:
        logging.exception(f"Job {job_id} failed: {e}")
        fail_key = f"jobs/failed/{job_id}.json"
        fail_payload = {
            "job_id": job_id,
            "mode": mode,
            "video_key": video_key,
            "status": "failed",
            "error": str(e),
        }
        put_json(AWS_BUCKET, fail_key, fail_payload)

        # ลองย้าย metadata ไป failed ด้วย
        try:
            move_s3_object(pending_meta_key, f"jobs/failed/{job_id}_meta.json")
        except Exception:
            logging.warning("Move pending meta to failed failed too")

# ----------------------------------------------------------
# Main loop
# ----------------------------------------------------------
def main_loop():
    logging.info("Worker started. Waiting for jobs...")
    logging.info(f"Bucket={AWS_BUCKET}, region={AWS_REGION}, poll_interval={JOB_POLL_INTERVAL}s")

    while True:
        try:
            job_ids = list_pending_job_ids()
            if not job_ids:
                logging.info("No pending jobs. Sleeping %s seconds...", JOB_POLL_INTERVAL)
                time.sleep(JOB_POLL_INTERVAL)
                continue

            logging.info(f"Found {len(job_ids)} pending job(s): {job_ids}")
            for job_id in job_ids:
                handle_job(job_id)

        except Exception as e:
            logging.exception(f"Error in main loop: {e}")
            time.sleep(JOB_POLL_INTERVAL)

if __name__ == "__main__":
    main_loop()
