import os
import json
import time
import logging
import tempfile
import subprocess
from datetime import datetime, timezone

import boto3

# Optional heavy libs – รองรับกรณี import ไม่ได้ ด้วยการ fail job สวย ๆ
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore
    MP_HAS_SOLUTIONS = hasattr(mp, "solutions")
except Exception:
    mp = None  # type: ignore
    MP_HAS_SOLUTIONS = False

# ---------------------------------------------------------------------------
# Config & logger
# ---------------------------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("worker")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"


# ---------------------------------------------------------------------------
# Small S3 helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> dict:
    logger.info("[s3_get_json] key=%s", key)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: dict) -> None:
    body_str = json.dumps(payload)
    logger.info("[s3_put_json] key=%s size=%d bytes", key, len(body_str))
    body = body_str.encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


def download_to_temp(key: str, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    logger.info("[s3_download] %s -> %s", key, path)
    with open(path, "wb") as f:
        s3.download_fileobj(AWS_BUCKET, key, f)
    return path


def upload_from_path(path: str, key: str, content_type: str = "video/mp4") -> None:
    logger.info("[s3_upload] %s -> %s", path, key)
    with open(path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, key, ExtraArgs={"ContentType": content_type})


def copy_video_in_s3(input_key: str, output_key: str) -> None:
    logger.info("[copy_object] %s -> %s", input_key, output_key)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": input_key},
        Key=output_key,
        ContentType="video/mp4",
    )


# ---------------------------------------------------------------------------
# Job lifecycle helpers
# ---------------------------------------------------------------------------

def list_pending_json_keys():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                yield key


def find_one_pending_job_key() -> str | None:
    for key in list_pending_json_keys():
        logger.info("[find_one_pending_job_key] found %s", key)
        return key
    logger.debug("[find_one_pending_job_key] no pending jobs")
    return None


def move_json(old_key: str, new_key: str, payload: dict) -> None:
    s3_put_json(new_key, payload)
    if old_key != new_key:
        logger.info("[s3_delete] key=%s", old_key)
        s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)


def update_status(job: dict, status: str, error: str | None = None) -> dict:
    job["status"] = status
    job["updated_at"] = utc_now_iso()
    if error is not None:
        job["error"] = error
    return job


# ---------------------------------------------------------------------------
# Audio helper (ffmpeg)
# ---------------------------------------------------------------------------

def _ffmpeg_exists() -> bool:
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def attach_audio_ffmpeg(video_no_audio_path: str, original_path: str, out_path: str) -> None:
    """
    เอา video stream จากไฟล์ overlay + เอา audio stream จากไฟล์ต้นฉบับ
    """
    if not _ffmpeg_exists():
        raise RuntimeError("ffmpeg not found. Cannot keep audio. (Install ffmpeg or set keep_audio=False)")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_no_audio_path,
        "-i", original_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    logger.info("[ffmpeg] attach audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------------------
# Dots (Johansson) processing
# ---------------------------------------------------------------------------

def process_dots_video(
    input_key: str,
    output_key: str,
    multi_person: bool = False,
    dot_px: int = 5,
    keep_audio: bool = True,
) -> None:
    """
    สร้างวิดีโอ Johansson dot:
      - อ่านวิดีโอจาก S3
      - ใช้ MediaPipe Pose หา joint
      - วาดจุดสีขาวลงบนพื้นหลังดำ
      - dot_px: 1–20
      - keep_audio: True/False (ใช้ ffmpeg ดึงเสียงจากต้นฉบับกลับเข้าไฟล์ output)
    """
    if cv2 is None or np is None or not (mp and MP_HAS_SOLUTIONS):
        raise RuntimeError("Johansson dots mode requires OpenCV, NumPy, and MediaPipe to be installed")

    dot_px = int(dot_px)
    dot_px = max(1, min(20, dot_px))

    input_path = download_to_temp(input_key, suffix=".mp4")

    # เราจะ render แบบ silent ก่อนเสมอ (cv2 ไม่มี audio)
    out_silent = tempfile.mktemp(suffix=".mp4")
    out_final = tempfile.mktemp(suffix=".mp4")

    logger.info(
        "[dots] start input=%s out_silent=%s dot_px=%s keep_audio=%s multi_person=%s",
        input_path,
        out_silent,
        dot_px,
        keep_audio,
        multi_person,
    )

    cap = cv2.VideoCapture(input_path)  # type: ignore[arg-type]
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Cannot read any frame from input video")
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_silent, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")

    pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.resize(frame, (width, height))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)  # type: ignore[arg-type]

                black = np.zeros((height, width, 3), dtype=np.uint8)

                if results.pose_landmarks:
                    h, w, _ = black.shape
                    for lm in results.pose_landmarks.landmark:
                        if getattr(lm, "visibility", 1.0) < 0.5:
                            continue
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(
                                black,
                                (x, y),
                                dot_px,  # ✅ ให้เลือก 1–20 px
                                (255, 255, 255),
                                -1,
                                lineType=cv2.LINE_AA,
                            )

                writer.write(black)

    finally:
        cap.release()
        writer.release()

    # ✅ ถ้าต้องการเสียง: ใช้ ffmpeg เอาเสียงจาก input_path มาใส่ out_silent
    try:
        if keep_audio:
            attach_audio_ffmpeg(out_silent, input_path, out_final)
            upload_from_path(out_final, output_key)
        else:
            upload_from_path(out_silent, output_key)
    finally:
        for path in (input_path, out_silent, out_final):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Job processor
# ---------------------------------------------------------------------------

def process_job(job_json_key: str) -> None:
    raw_job = s3_get_json(job_json_key)

    job_id = raw_job.get("job_id")
    mode = raw_job.get("mode", "passthrough")
    input_key = raw_job.get("input_key")

    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")
    if not input_key:
        raise ValueError("Job JSON missing 'input_key'")

    output_key = raw_job.get("output_key") or f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

    # ✅ NEW: params from app
    params = raw_job.get("params", {}) if isinstance(raw_job, dict) else {}
    dot_px = int(params.get("dot_px", 5))
    dot_px = max(1, min(20, dot_px))
    keep_audio = bool(params.get("keep_audio", True))

    logger.info(
        "[process_job] job_id=%s mode=%s input_key=%s output_key=%s dot_px=%s keep_audio=%s",
        job_id,
        mode,
        input_key,
        output_key,
        dot_px,
        keep_audio,
    )

    job = dict(raw_job)
    job = update_status(job, "processing")
    job["output_key"] = output_key

    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_json(job_json_key, processing_key, job)

    try:
        if mode in ("dots", "dots_1p", "dots_single"):
            process_dots_video(
                input_key,
                output_key,
                multi_person=False,
                dot_px=dot_px,
                keep_audio=keep_audio,
            )
        elif mode in ("dots_2p", "dots_multi"):
            process_dots_video(
                input_key,
                output_key,
                multi_person=True,
                dot_px=dot_px,
                keep_audio=keep_audio,
            )
        else:
            # passthrough/copy (มีเสียงเดิมอยู่แล้ว)
            copy_video_in_s3(input_key, output_key)

        job = update_status(job, "finished", error=None)
        finished_key = f"{FINISHED_PREFIX}/{job_id}.json"
        move_json(processing_key, finished_key, job)
        logger.info("[process_job] job_id=%s finished", job_id)

    except Exception as exc:
        logger.exception("[process_job] job_id=%s FAILED: %s", job_id, exc)
        job = update_status(job, "failed", error=str(exc))
        failed_key = f"{FAILED_PREFIX}/{job_id}.json"
        move_json(processing_key, failed_key, job)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("====== AI People Reader Worker (Johansson) ======")
    logger.info("Using bucket: %s", AWS_BUCKET)
    logger.info("Region       : %s", AWS_REGION)
    logger.info("Poll every   : %s seconds", POLL_INTERVAL)
    logger.info("MP available : %s", bool(mp and MP_HAS_SOLUTIONS))
    logger.info("cv2 available: %s", cv2 is not None)
    logger.info("numpy avail. : %s", np is not None)
    logger.info("ffmpeg avail.: %s", _ffmpeg_exists())

    while True:
        try:
            job_key = find_one_pending_job_key()
            if job_key:
                process_job(job_key)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            logger.exception("[main] Unexpected error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
