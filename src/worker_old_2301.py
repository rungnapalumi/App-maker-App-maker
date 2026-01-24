# worker.py — AI People Reader Worker (Dots + Skeleton)
#
# ✅ Modes:
#   - dots     : Johansson dots on BLACK background (no audio)
#   - skeleton : Skeleton overlay on REAL video (no audio)
#   - clear    : Copy input -> output (no processing)
#
# ✅ Params supported (from job JSON):
#   - dot_radius: int (1–20)
#   - skeleton_color: str hex เช่น "#00FF00" (default green)
#   - skeleton_thickness: int (default 2)
#
# Backward compatible:
#   - Accept params in job["params"] or at top-level keys.
#
# IMPORTANT:
#   This worker REQUIRES these libraries installed in worker environment:
#     - boto3
#     - python-dotenv (optional)
#     - opencv-python-headless
#     - numpy
#     - mediapipe
#
# If you see "requires OpenCV, NumPy, and MediaPipe", your worker_requirements.txt is missing them.

import os
import json
import time
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import boto3

# Optional heavy libs — if missing, we fail job nicely
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


# -----------------------------------------------------------------------------
# Config & logger
# -----------------------------------------------------------------------------
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
PENDING_PREFIX = f"{JOBS_PREFIX}/pending/"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing/"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished/"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed/"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output/"

# MediaPipe Pose landmark indices 0–10 are face-related in Pose
FACE_LMS = set(range(0, 11))

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _require_cv_stack(context: str) -> None:
    if cv2 is None or np is None or not (mp and MP_HAS_SOLUTIONS):
        raise RuntimeError(
            f"{context} requires OpenCV, NumPy, and MediaPipe to be installed"
        )

def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(v)
    except Exception:
        return default
    return max(lo, min(hi, iv))

def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    Input: '#RRGGBB' or 'RRGGBB' (case-insensitive)
    Output: (B, G, R) for OpenCV
    """
    if not hex_color:
        return (0, 255, 0)
    s = hex_color.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        return (0, 255, 0)
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)
    except Exception:
        return (0, 255, 0)

def s3_get_json(key: str) -> Dict[str, Any]:
    logger.info("[s3_get_json] key=%s", key)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))

def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body_str = json.dumps(payload, ensure_ascii=False)
    logger.info("[s3_put_json] key=%s size=%d bytes", key, len(body_str))
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body_str.encode("utf-8"),
        ContentType="application/json",
    )

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
        s3.upload_fileobj(
            f,
            AWS_BUCKET,
            key,
            ExtraArgs={"ContentType": content_type},
        )

def copy_video_in_s3(input_key: str, output_key: str) -> None:
    logger.info("[copy_object] %s -> %s", input_key, output_key)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": input_key},
        Key=output_key,
        ContentType="video/mp4",
        MetadataDirective="REPLACE",
    )

# -----------------------------------------------------------------------------
# Job lifecycle helpers
# -----------------------------------------------------------------------------
def list_pending_json_keys():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX):
        for item in page.get("Contents", []):
            key = item["Key"]
            # pending job json is: jobs/pending/<job_id>.json
            if key.endswith(".json") and key.count("/") == PENDING_PREFIX.count("/"):
                yield key

def find_one_pending_job_key() -> Optional[str]:
    for key in list_pending_json_keys():
        logger.info("[find_one_pending_job_key] found %s", key)
        return key
    return None

def move_json(old_key: str, new_key: str, payload: Dict[str, Any]) -> None:
    s3_put_json(new_key, payload)
    if old_key != new_key:
        logger.info("[s3_delete] key=%s", old_key)
        s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)

def update_status(job: Dict[str, Any], status: str, error: Optional[str] = None) -> Dict[str, Any]:
    job["status"] = status
    job["updated_at"] = utc_now_iso()
    if error is not None:
        job["error"] = error
    return job

def get_param(job: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Read param from:
      - job["params"][key]
      - or job[key] (backward compatibility)
    """
    params = job.get("params")
    if isinstance(params, dict) and key in params:
        return params.get(key)
    return job.get(key, default)

# -----------------------------------------------------------------------------
# Video processing: Dots
# -----------------------------------------------------------------------------
def process_dots_video(input_key: str, output_key: str, dot_radius: int) -> None:
    """
    Create Johansson dots on BLACK background.
    NO AUDIO.
    """
    _require_cv_stack("Johansson dots mode")

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_path = tempfile.mktemp(suffix=".mp4")

    cap = cv2.VideoCapture(input_path)  # type: ignore[arg-type]
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Cannot read any frame from input video")
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
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

    logger.info("[dots] radius=%s input=%s out=%s", dot_radius, input_path, out_path)

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
                                int(dot_radius),
                                (255, 255, 255),
                                -1,
                                lineType=cv2.LINE_8,  # sharp
                            )

                writer.write(black)

    finally:
        cap.release()
        writer.release()
        try:
            upload_from_path(out_path, output_key)
        finally:
            for p in (input_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

# -----------------------------------------------------------------------------
# Video processing: Skeleton overlay (BODY ONLY, no face)
# -----------------------------------------------------------------------------
def process_skeleton_video(
    input_key: str,
    output_key: str,
    skeleton_color_hex: str,
    skeleton_thickness: int,
) -> None:
    """
    Overlay skeleton on REAL video.
    BODY ONLY: remove face lines/landmarks (0–10).
    NO AUDIO.
    """
    _require_cv_stack("Skeleton mode")

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_path = tempfile.mktemp(suffix=".mp4")

    cap = cv2.VideoCapture(input_path)  # type: ignore[arg-type]
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open input video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Cannot read any frame from input video")
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")

    color_bgr = _hex_to_bgr(skeleton_color_hex)
    thickness = max(1, int(skeleton_thickness))

    pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    logger.info(
        "[skeleton] color=%s bgr=%s thickness=%s input=%s out=%s",
        skeleton_color_hex,
        color_bgr,
        thickness,
        input_path,
        out_path,
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

                if results.pose_landmarks:
                    h, w, _ = frame.shape

                    # Build landmark map but remove face landmarks completely
                    lm_xy: Dict[int, Optional[Tuple[int, int]]] = {}
                    for idx, lm in enumerate(results.pose_landmarks.landmark):
                        if idx in FACE_LMS:
                            lm_xy[idx] = None
                            continue
                        if getattr(lm, "visibility", 1.0) < 0.5:
                            lm_xy[idx] = None
                            continue
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            lm_xy[idx] = (x, y)
                        else:
                            lm_xy[idx] = None

                    # Draw connections (BODY ONLY) — sharp lines
                    for a, b in mp.solutions.pose.POSE_CONNECTIONS:  # type: ignore[attr-defined]
                        if a in FACE_LMS or b in FACE_LMS:
                            continue
                        pa = lm_xy.get(a)
                        pb = lm_xy.get(b)
                        if pa and pb:
                            cv2.line(
                                frame,
                                pa,
                                pb,
                                color_bgr,
                                thickness,
                                lineType=cv2.LINE_8,  # << sharp
                            )

                writer.write(frame)

    finally:
        cap.release()
        writer.release()
        try:
            upload_from_path(out_path, output_key)
        finally:
            for p in (input_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

# -----------------------------------------------------------------------------
# Job processor
# -----------------------------------------------------------------------------
def process_job(job_json_key: str) -> None:
    raw_job = s3_get_json(job_json_key)

    job_id = raw_job.get("job_id")
    mode = (raw_job.get("mode") or "clear").strip().lower()
    input_key = raw_job.get("input_key")

    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")
    if not input_key:
        raise ValueError("Job JSON missing 'input_key'")

    output_key = raw_job.get("output_key") or f"{OUTPUT_PREFIX}{job_id}/result.mp4"

    # Parameters (support both params dict and top-level keys)
    dot_radius = _clamp_int(get_param(raw_job, "dot_radius", get_param(raw_job, "dot_px", 5)), 1, 20, 5)
    skeleton_color = str(get_param(raw_job, "skeleton_color", "#00FF00") or "#00FF00")
    skeleton_thickness = _clamp_int(get_param(raw_job, "skeleton_thickness", 2), 1, 12, 2)

    logger.info(
        "[process_job] job_id=%s mode=%s input_key=%s output_key=%s",
        job_id,
        mode,
        input_key,
        output_key,
    )

    # Move JSON -> processing
    job = dict(raw_job)
    job["output_key"] = output_key
    job = update_status(job, "processing")

    processing_key = f"{PROCESSING_PREFIX}{job_id}.json"
    move_json(job_json_key, processing_key, job)

    try:
        if mode == "dots":
            process_dots_video(input_key, output_key, dot_radius=dot_radius)

        elif mode == "skeleton":
            process_skeleton_video(
                input_key,
                output_key,
                skeleton_color_hex=skeleton_color,
                skeleton_thickness=skeleton_thickness,
            )

        else:
            # clear / passthrough
            copy_video_in_s3(input_key, output_key)

        job = update_status(job, "finished", error=None)
        finished_key = f"{FINISHED_PREFIX}{job_id}.json"
        move_json(processing_key, finished_key, job)
        logger.info("[process_job] job_id=%s finished", job_id)

    except Exception as exc:
        logger.exception("[process_job] job_id=%s FAILED: %s", job_id, exc)
        job = update_status(job, "failed", error=str(exc))
        failed_key = f"{FAILED_PREFIX}{job_id}.json"
        move_json(processing_key, failed_key, job)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
def main() -> None:
    logger.info("WORKER VERSION: final_clean_no_audio_body_only_face_removed")
    logger.info("====== AI People Reader Worker (No Audio) ======")
    logger.info("Using bucket : %s", AWS_BUCKET)
    logger.info("Region      : %s", AWS_REGION)
    logger.info("Poll every  : %s seconds", POLL_INTERVAL)
    logger.info("MP available: %s", bool(mp and MP_HAS_SOLUTIONS))
    logger.info("cv2 avail.  : %s", cv2 is not None)
    logger.info("numpy avail.: %s", np is not None)

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
