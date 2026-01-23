# worker.py — AI People Reader Worker (dots + skeleton OVERLAY on original video + keep audio via MoviePy)
# ✅ mode=dots: Johansson dots on BLACK background
# ✅ mode=skeleton: skeleton OVERLAY on ORIGINAL video (not black)
# ✅ keep_audio: attach original audio AFTER processing (MoviePy)
# ✅ dot_px: 1–20 via params.dot_px
# ✅ If skeleton detection is too low -> FAIL clearly (prevents "looks like input" confusion)

import os
import json
import time
import logging
import tempfile
from datetime import datetime, timezone

import boto3
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip

# ---------------------------------------------------------------------------
# Config & logger
# ---------------------------------------------------------------------------

AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET / S3_BUCKET")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("worker")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING_PREFIX = "jobs/pending"
PROCESSING_PREFIX = "jobs/processing"
FINISHED_PREFIX = "jobs/finished"
FAILED_PREFIX = "jobs/failed"
OUTPUT_PREFIX = "jobs/output"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> dict:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(key: str, payload: dict) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def download_temp(key: str, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        s3.download_fileobj(AWS_BUCKET, key, f)
    return path


def upload_video(path: str, key: str) -> None:
    with open(path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})


# ---------------------------------------------------------------------------
# Audio (MoviePy – reliable on Render)
# ---------------------------------------------------------------------------

def attach_audio_moviepy(silent_video_path: str, original_video_path: str, out_path: str) -> None:
    """
    Take processed video (no audio) and attach original audio track.
    MoviePy uses imageio-ffmpeg under the hood (works well on Render).
    """
    silent_clip = VideoFileClip(silent_video_path)
    original_clip = VideoFileClip(original_video_path)

    if original_clip.audio is None:
        silent_clip.close()
        original_clip.close()
        raise RuntimeError("Original video has NO audio track (cannot keep_audio=True)")

    final_clip = silent_clip.set_audio(original_clip.audio)

    final_clip.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        fps=silent_clip.fps,
        verbose=False,
        logger=None,
    )

    silent_clip.close()
    original_clip.close()
    final_clip.close()


# ---------------------------------------------------------------------------
# Video processors (silent first)
# ---------------------------------------------------------------------------

def _open_video_and_meta(input_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Fallback if metadata broken
    if w <= 0 or h <= 0:
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Cannot read any frame from input video")
        h, w = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return cap, w, h, fps


def render_dots_silent(input_path: str, out_path: str, dot_px: int) -> None:
    cap, w, h, fps = _open_video_and_meta(input_path)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")

    dot_px = max(1, min(20, int(dot_px)))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    if getattr(lm, "visibility", 1.0) < 0.5:
                        continue
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(canvas, (x, y), dot_px, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            writer.write(canvas)

    cap.release()
    writer.release()


def render_skeleton_overlay_silent(input_path: str, out_path: str) -> None:
    """
    Skeleton OVERLAY on the original video (so teacher sees body + skeleton).
    If pose detection is too low, FAIL clearly so you don't get "looks like input".
    """
    logger.info("[skeleton] START overlay render")

    cap, w, h, fps = _open_video_and_meta(input_path)

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")

    # Make overlay very visible (white, thick)
    landmark_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=3)
    connection_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=2)

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    total_frames = 0
    detected_frames = 0

    with pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total_frames += 1

            frame = cv2.resize(frame, (w, h))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            canvas = frame.copy()  # ✅ overlay on original

            if res.pose_landmarks:
                detected_frames += 1
                mp_draw.draw_landmarks(
                    canvas,
                    res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )

            writer.write(canvas)

    cap.release()
    writer.release()

    ratio = (detected_frames / total_frames) if total_frames else 0.0
    logger.info("[skeleton] detected %d/%d (%.1f%%)", detected_frames, total_frames, ratio * 100.0)

    # If detection is too low, output would look like input -> fail loudly
    if total_frames == 0 or ratio < 0.10:
        raise RuntimeError(
            f"Skeleton overlay not detected enough (pose found in {detected_frames}/{total_frames} frames). "
            "Try: full body visible, better lighting, less motion blur, camera not too far."
        )


# ---------------------------------------------------------------------------
# Job handling
# ---------------------------------------------------------------------------

def find_pending_job_key() -> str | None:
    r = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX)
    for o in r.get("Contents", []):
        if o["Key"].endswith(".json"):
            return o["Key"]
    return None


def move_job_json(old_key: str, new_key: str, payload: dict) -> None:
    s3_put_json(new_key, payload)
    s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)


def process_job(job_json_key: str) -> None:
    job = s3_get_json(job_json_key)

    job_id = job.get("job_id")
    mode = job.get("mode", "dots")
    input_key = job.get("input_key")

    if not job_id:
        raise RuntimeError("Job JSON missing job_id")
    if not input_key:
        raise RuntimeError("Job JSON missing input_key")

    params = job.get("params", {})
    if not isinstance(params, dict):
        params = {}

    dot_px = max(1, min(20, int(params.get("dot_px", 5))))
    keep_audio = bool(params.get("keep_audio", True))

    output_key = job.get("output_key") or f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

    logger.info("[job] id=%s mode=%s keep_audio=%s dot_px=%s", job_id, mode, keep_audio, dot_px)

    # Move to processing
    job["status"] = "processing"
    job["updated_at"] = utc_now_iso()
    job["output_key"] = output_key

    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_job_json(job_json_key, processing_key, job)

    input_path = None
    silent_path = None
    final_path = None

    try:
        input_path = download_temp(input_key)
        silent_path = tempfile.mktemp(suffix=".mp4")
        final_path = tempfile.mktemp(suffix=".mp4")

        if mode == "dots":
            render_dots_silent(input_path, silent_path, dot_px)
        elif mode == "skeleton":
            render_skeleton_overlay_silent(input_path, silent_path)
        else:
            raise RuntimeError(f"Unsupported mode: {mode}")

        if keep_audio:
            attach_audio_moviepy(silent_path, input_path, final_path)
            upload_video(final_path, output_key)
        else:
            upload_video(silent_path, output_key)

        job["status"] = "finished"
        job["updated_at"] = utc_now_iso()
        move_job_json(processing_key, f"{FINISHED_PREFIX}/{job_id}.json", job)
        logger.info("[job] finished id=%s", job_id)

    except Exception as e:
        logger.exception("[job] FAILED id=%s", job_id)
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = utc_now_iso()
        move_job_json(processing_key, f"{FAILED_PREFIX}/{job_id}.json", job)

    finally:
        for p in (input_path, silent_path, final_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== AI People Reader Worker START (dots + skeleton overlay + audio) ===")
    while True:
        try:
            job_key = find_pending_job_key()
            if job_key:
                process_job(job_key)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception:
            logger.exception("Worker loop error")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
