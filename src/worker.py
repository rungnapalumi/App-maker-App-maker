# worker.py — AI People Reader Worker (Johansson dots, audio-safe)

import os
import json
import time
import logging
import tempfile
import subprocess
from datetime import datetime, timezone

import boto3

# Optional heavy libs
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
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("worker")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING_PREFIX = "jobs/pending"
PROCESSING_PREFIX = "jobs/processing"
FINISHED_PREFIX = "jobs/finished"
FAILED_PREFIX = "jobs/failed"
OUTPUT_PREFIX = "jobs/output"


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bin_exists(name: str) -> bool:
    try:
        return subprocess.run(
            [name, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except Exception:
        return False


def ffmpeg_exists() -> bool:
    return _bin_exists("ffmpeg")


def ffprobe_exists() -> bool:
    return _bin_exists("ffprobe")


def has_audio(path: str) -> bool:
    if not ffprobe_exists():
        logger.warning("ffprobe not found; assuming audio exists")
        return True

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    result = bool((r.stdout or "").strip())
    logger.info("[ffprobe] audio=%s file=%s", result, path)
    return result


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_get_json(key: str) -> dict:
    return json.loads(
        s3.get_object(Bucket=AWS_BUCKET, Key=key)["Body"].read().decode("utf-8")
    )


def s3_put_json(key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def download_temp(key: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(path, "wb") as f:
        s3.download_fileobj(AWS_BUCKET, key, f)
    return path


def upload_video(path: str, key: str) -> None:
    with open(path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})


# ---------------------------------------------------------------------------
# ffmpeg audio pipeline (GUARANTEED)
# ---------------------------------------------------------------------------

def extract_audio(original: str, audio_out: str) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not installed")

    cmd = [
        "ffmpeg", "-y",
        "-i", original,
        "-vn",
        "-c:a", "aac",
        "-b:a", "192k",
        audio_out,
    ]
    logger.info("[ffmpeg] extract audio")
    subprocess.run(cmd, check=True)


def mux_audio(video: str, audio: str, out: str) -> None:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not installed")

    cmd = [
        "ffmpeg", "-y",
        "-i", video,
        "-i", audio,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-c:a", "aac",
        "-shortest",
        "-movflags", "+faststart",
        out,
    ]
    logger.info("[ffmpeg] mux audio")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Dots processing
# ---------------------------------------------------------------------------

def process_dots(input_key: str, output_key: str, dot_px: int, keep_audio: bool):
    if cv2 is None or np is None or not MP_HAS_SOLUTIONS:
        raise RuntimeError("OpenCV / NumPy / MediaPipe missing")

    input_path = download_temp(input_key)
    silent_out = tempfile.mktemp(suffix=".mp4")
    final_out = tempfile.mktemp(suffix=".mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    writer = cv2.VideoWriter(
        silent_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    pose = mp.solutions.pose.Pose()

    with pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    if lm.visibility < 0.5:
                        continue
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(canvas, (x, y), dot_px, (255, 255, 255), -1)

            writer.write(canvas)

    cap.release()
    writer.release()

    # ---------- AUDIO PIPELINE ----------
    if keep_audio:
        logger.info("[audio] keep_audio=True")

        if not has_audio(input_path):
            raise RuntimeError("Original video has NO audio stream")

        audio_path = tempfile.mktemp(suffix=".m4a")
        extract_audio(input_path, audio_path)
        mux_audio(silent_out, audio_path, final_out)

        if not has_audio(final_out):
            raise RuntimeError("Audio mux FAILED — output still silent")

        upload_video(final_out, output_key)
    else:
        upload_video(silent_out, output_key)

    for p in (input_path, silent_out, final_out):
        if p and os.path.exists(p):
            os.remove(p)


# ---------------------------------------------------------------------------
# Job handling
# ---------------------------------------------------------------------------

def find_pending_job():
    r = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    for o in r.get("Contents", []):
        if o["Key"].endswith(".json"):
            return o["Key"]
    return None


def move_job(old: str, new: str, payload: dict):
    s3_put_json(new, payload)
    s3.delete_object(Bucket=AWS_BUCKET, Key=old)


def process_job(job_key: str):
    job = s3_get_json(job_key)
    job_id = job["job_id"]

    params = job.get("params", {})
    dot_px = int(params.get("dot_px", 5))
    keep_audio = bool(params.get("keep_audio", True))

    input_key = job["input_key"]
    output_key = job.get("output_key") or f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

    job["status"] = "processing"
    job["updated_at"] = utc_now_iso()

    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_job(job_key, processing_key, job)

    try:
        process_dots(input_key, output_key, dot_px, keep_audio)
        job["status"] = "finished"
        job["updated_at"] = utc_now_iso()
        move_job(processing_key, f"{FINISHED_PREFIX}/{job_id}.json", job)
    except Exception as e:
        logger.exception("JOB FAILED")
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = utc_now_iso()
        move_job(processing_key, f"{FAILED_PREFIX}/{job_id}.json", job)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    logger.info("=== AI People Reader Worker START ===")
    logger.info("ffmpeg=%s ffprobe=%s", ffmpeg_exists(), ffprobe_exists())

    while True:
        try:
            job_key = find_pending_job()
            if job_key:
                process_job(job_key)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.exception("Worker loop error")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
