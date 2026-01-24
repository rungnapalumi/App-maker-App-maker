# worker.py — AI People Reader Worker (Johansson dots + Skeleton overlay + audio merge)
#
# ✅ Features
# - dots (Johansson): black background + white dots, configurable radius (1–20)
# - skeleton: overlay on REAL input video (NOT copy), configurable line color + thickness
# - keep_audio: merges original audio back into processed output using ffmpeg (if available)
#
# ✅ Job JSON supported fields (optional)
# {
#   "mode": "dots" | "skeleton" | "passthrough",
#   "keep_audio": true/false,
#   "dot_radius": 1..20,
#   "skeleton_line_color": "#00FF00" | "green" | [0,255,0] | [r,g,b],
#   "skeleton_line_thickness": 1..20
# }
#
# Defaults:
# - keep_audio = False
# - dot_radius = 5
# - skeleton_line_color = green (#00FF00)
# - skeleton_line_thickness = 2
#
# Notes:
# - OpenCV cannot write audio. We render video first, then (optionally) merge audio via ffmpeg.
# - If ffmpeg is missing or audio stream absent, job still finishes with silent processed video.

import os
import json
import time
import logging
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import boto3

# Optional heavy libs – graceful fail if missing
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


def find_one_pending_job_key() -> Optional[str]:
    for key in list_pending_json_keys():
        logger.info("[find_one_pending_job_key] found %s", key)
        return key
    logger.debug("[find_one_pending_job_key] no pending jobs")
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


# ---------------------------------------------------------------------------
# ffmpeg helpers (audio merge)
# ---------------------------------------------------------------------------


def _has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def merge_audio_from_input(
    video_no_audio_path: str,
    original_input_path: str,
    out_path: str,
) -> bool:
    """
    Merge audio from original_input into processed video.
    Returns True if merge succeeded, else False (keeps silent video).
    """
    if not _has_ffmpeg():
        logger.warning("[ffmpeg] ffmpeg not found. Output will be silent.")
        return False

    # Map video from processed (0:v) and audio from original (1:a)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_no_audio_path,
        "-i",
        original_input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]

    try:
        logger.info("[ffmpeg] merging audio: %s", " ".join(cmd))
        r = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            # Common cause: input has no audio stream
            logger.warning("[ffmpeg] merge failed (rc=%s). stderr tail: %s", r.returncode, r.stderr[-600:])
            return False
        return True
    except Exception as exc:
        logger.warning("[ffmpeg] merge exception: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Param parsing
# ---------------------------------------------------------------------------


def clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        iv = int(v)
    except Exception:
        return default
    return max(lo, min(hi, iv))


def parse_color(color_value: Any) -> Tuple[int, int, int]:
    """
    Returns BGR tuple for OpenCV.
    Accepted:
      - "#RRGGBB"
      - "green"/"red"/"blue"/"white"/"black"/"yellow"/"cyan"/"magenta"
      - [r,g,b] or [b,g,r]
      - {"r":..,"g":..,"b":..}
    Default: green (#00FF00)
    """
    # Default green (RGB 0,255,0) -> BGR (0,255,0)
    default_bgr = (0, 255, 0)

    if color_value is None:
        return default_bgr

    # Named colors (RGB)
    named = {
        "green": (0, 255, 0),
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }

    try:
        # string
        if isinstance(color_value, str):
            s = color_value.strip().lower()
            if s in named:
                r, g, b = named[s]
                return (b, g, r)

            if s.startswith("#") and len(s) == 7:
                r = int(s[1:3], 16)
                g = int(s[3:5], 16)
                b = int(s[5:7], 16)
                return (b, g, r)

            # "r,g,b"
            if "," in s:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) == 3:
                    a, b, c = [int(x) for x in parts]
                    # assume RGB
                    return (c, b, a)

        # dict
        if isinstance(color_value, dict):
            r = int(color_value.get("r", 0))
            g = int(color_value.get("g", 255))
            b = int(color_value.get("b", 0))
            return (b, g, r)

        # list/tuple
        if isinstance(color_value, (list, tuple)) and len(color_value) == 3:
            a, b, c = [int(x) for x in color_value]
            # assume RGB by default
            return (c, b, a)

    except Exception:
        return default_bgr

    return default_bgr


# ---------------------------------------------------------------------------
# Core video processing
# ---------------------------------------------------------------------------


def _require_cv_stack(feature: str) -> None:
    if cv2 is None or np is None or not (mp and MP_HAS_SOLUTIONS):
        raise RuntimeError(
            f"{feature} requires OpenCV, NumPy, and MediaPipe to be installed"
        )


def _open_video_capture(path: str):
    cap = cv2.VideoCapture(path)  # type: ignore[arg-type]
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open input video")
    return cap


def _read_video_meta(cap) -> Tuple[int, int, float]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if width <= 0 or height <= 0:
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("Cannot read any frame from input video")
        height, width = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return width, height, float(fps)


def _make_writer(out_path: str, width: int, height: int, fps: float):
    # mp4v is widely compatible (QuickTime-friendly)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")
    return writer


def process_dots_video(
    input_key: str,
    output_key: str,
    *,
    dot_radius: int,
    keep_audio: bool,
) -> None:
    """
    Johansson dots: black background + white dots.
    If keep_audio=True, merge audio back.
    """
    _require_cv_stack("Johansson dots mode")

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_no_audio = tempfile.mktemp(suffix=".mp4")
    out_final = tempfile.mktemp(suffix=".mp4")

    logger.info(
        "[dots] start input=%s out_no_audio=%s radius=%s keep_audio=%s",
        input_path,
        out_no_audio,
        dot_radius,
        keep_audio,
    )

    cap = _open_video_capture(input_path)
    width, height, fps = _read_video_meta(cap)
    writer = _make_writer(out_no_audio, width, height, fps)

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
                                dot_radius,
                                (255, 255, 255),
                                -1,
                                lineType=cv2.LINE_8,  # sharp
                            )

                writer.write(black)

    finally:
        cap.release()
        writer.release()

    # Audio merge (optional)
    produced_path = out_no_audio
    if keep_audio:
        merged_ok = merge_audio_from_input(out_no_audio, input_path, out_final)
        if merged_ok:
            produced_path = out_final
        else:
            logger.warning("[dots] keep_audio requested but merge failed; output silent.")

    try:
        upload_from_path(produced_path, output_key, content_type="video/mp4")
    finally:
        for p in (input_path, out_no_audio, out_final):
            try:
                os.remove(p)
            except OSError:
                pass


def process_skeleton_video(
    input_key: str,
    output_key: str,
    *,
    line_bgr: Tuple[int, int, int],
    thickness: int,
    keep_audio: bool,
) -> None:
    """
    Skeleton overlay on REAL video frames.
    - Default line color: green
    - Sharp line: cv2.LINE_8
    - If keep_audio=True, merge audio back
    """
    _require_cv_stack("Skeleton mode")

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_no_audio = tempfile.mktemp(suffix=".mp4")
    out_final = tempfile.mktemp(suffix=".mp4")

    logger.info(
        "[skeleton] start input=%s out_no_audio=%s color(BGR)=%s thickness=%s keep_audio=%s",
        input_path,
        out_no_audio,
        line_bgr,
        thickness,
        keep_audio,
    )

    cap = _open_video_capture(input_path)
    width, height, fps = _read_video_meta(cap)
    writer = _make_writer(out_no_audio, width, height, fps)

    pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    connections = mp.solutions.pose.POSE_CONNECTIONS  # type: ignore[attr-defined]

    def draw_skeleton(frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]

        # Convert landmarks to pixel points first
        pts = []
        for lm in landmarks.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            v = getattr(lm, "visibility", 1.0)
            pts.append((x, y, v))

        # Draw lines
        for a, b in connections:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < 0.5 or vb < 0.5:
                continue
            if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
                cv2.line(
                    frame_bgr,
                    (xa, ya),
                    (xb, yb),
                    line_bgr,
                    thickness,
                    lineType=cv2.LINE_8,  # sharp
                )

        # Optional: joints (small, sharp)
        joint_r = max(1, thickness)
        for (x, y, v) in pts:
            if v < 0.5:
                continue
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame_bgr, (x, y), joint_r, line_bgr, -1, lineType=cv2.LINE_8)

        return frame_bgr

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
                    frame = draw_skeleton(frame, results.pose_landmarks)

                writer.write(frame)

    finally:
        cap.release()
        writer.release()

    produced_path = out_no_audio
    if keep_audio:
        merged_ok = merge_audio_from_input(out_no_audio, input_path, out_final)
        if merged_ok:
            produced_path = out_final
        else:
            logger.warning("[skeleton] keep_audio requested but merge failed; output silent.")

    try:
        upload_from_path(produced_path, output_key, content_type="video/mp4")
    finally:
        for p in (input_path, out_no_audio, out_final):
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Job processor
# ---------------------------------------------------------------------------


def process_job(job_json_key: str) -> None:
    raw_job = s3_get_json(job_json_key)

    job_id = raw_job.get("job_id")
    mode = (raw_job.get("mode") or "passthrough").strip().lower()
    input_key = raw_job.get("input_key")

    if not job_id:
        raise ValueError("Job JSON missing 'job_id'")
    if not input_key:
        raise ValueError("Job JSON missing 'input_key'")

    output_key = raw_job.get("output_key") or f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

    # Options
    keep_audio = bool(raw_job.get("keep_audio", False))

    dot_radius = clamp_int(raw_job.get("dot_radius", 5), default=5, lo=1, hi=20)

    line_color_bgr = parse_color(raw_job.get("skeleton_line_color", "#00FF00"))
    line_thickness = clamp_int(raw_job.get("skeleton_line_thickness", 2), default=2, lo=1, hi=20)

    logger.info(
        "[process_job] job_id=%s mode=%s input_key=%s output_key=%s keep_audio=%s",
        job_id,
        mode,
        input_key,
        output_key,
        keep_audio,
    )

    # Move JSON to processing
    job = dict(raw_job)
    job["output_key"] = output_key
    job = update_status(job, "processing")

    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_json(job_json_key, processing_key, job)

    try:
        if mode in ("dots", "dots_1p", "dots_single", "johansson"):
            process_dots_video(
                input_key,
                output_key,
                dot_radius=dot_radius,
                keep_audio=keep_audio,
            )

        elif mode in ("skeleton", "pose", "skeleton_overlay"):
            process_skeleton_video(
                input_key,
                output_key,
                line_bgr=line_color_bgr,
                thickness=line_thickness,
                keep_audio=keep_audio,
            )

        else:
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
    logger.info("====== AI People Reader Worker (Dots + Skeleton) ======")
    logger.info("Using bucket: %s", AWS_BUCKET)
    logger.info("Region       : %s", AWS_REGION)
    logger.info("Poll every   : %s seconds", POLL_INTERVAL)
    logger.info("ffmpeg       : %s", _has_ffmpeg())
    logger.info("MP available : %s", bool(mp and MP_HAS_SOLUTIONS))
    logger.info("cv2 available: %s", cv2 is not None)
    logger.info("numpy avail. : %s", np is not None)

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
