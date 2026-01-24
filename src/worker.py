# worker.py — AI People Reader Worker (dots + skeleton overlay) | NO AUDIO
#
# ✅ dots: black background + white dots, dot_radius 1–20
# ✅ skeleton: overlay skeleton on original video (sharp lines)
# ✅ selectable skeleton line color + thickness (default green, thickness 2)
# ✅ If mode unknown -> passthrough copy
#
# Requires: boto3, numpy, opencv-python-headless, mediapipe

import os
import json
import time
import logging
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import boto3

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("worker")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def download_to_temp(key: str, suffix: str = ".mp4") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        s3.download_fileobj(AWS_BUCKET, key, f)
    return path


def upload_from_path(path: str, key: str) -> None:
    with open(path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, key, ExtraArgs={"ContentType": "video/mp4"})


def copy_video_in_s3(input_key: str, output_key: str) -> None:
    logger.info("[copy_object] %s -> %s", input_key, output_key)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        CopySource={"Bucket": AWS_BUCKET, "Key": input_key},
        Key=output_key,
        ContentType="video/mp4",
        MetadataDirective="REPLACE",
    )


def list_pending_json_keys():
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=PENDING_PREFIX):
        for item in page.get("Contents", []):
            k = item["Key"]
            if k.endswith(".json"):
                yield k


def find_one_pending_job_key() -> Optional[str]:
    for k in list_pending_json_keys():
        return k
    return None


def move_json(old_key: str, new_key: str, payload: Dict[str, Any]) -> None:
    s3_put_json(new_key, payload)
    if old_key != new_key:
        s3.delete_object(Bucket=AWS_BUCKET, Key=old_key)


def update_status(job: Dict[str, Any], status: str, error: Optional[str] = None) -> Dict[str, Any]:
    job["status"] = status
    job["updated_at"] = utc_now_iso()
    if error is not None:
        job["error"] = error
    return job


def clamp_int(v: Any, default: int, lo: int, hi: int) -> int:
    try:
        iv = int(v)
    except Exception:
        return default
    return max(lo, min(hi, iv))


def parse_color_to_bgr(color_value: Any) -> Tuple[int, int, int]:
    """
    Accept:
      - "#RRGGBB"
      - "green"/"red"/"blue"/"white"/"black"/"yellow"/"cyan"/"magenta"
      - "r,g,b"
      - [r,g,b]
      - {"r":..,"g":..,"b":..}
    Default: green (#00FF00)
    Returns BGR tuple for OpenCV.
    """
    default_bgr = (0, 255, 0)

    if color_value is None:
        return default_bgr

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

            if "," in s:
                parts = [p.strip() for p in s.split(",")]
                if len(parts) == 3:
                    r, g, b = [int(x) for x in parts]
                    return (b, g, r)

        if isinstance(color_value, dict):
            r = int(color_value.get("r", 0))
            g = int(color_value.get("g", 255))
            b = int(color_value.get("b", 0))
            return (b, g, r)

        if isinstance(color_value, (list, tuple)) and len(color_value) == 3:
            r, g, b = [int(x) for x in color_value]
            return (b, g, r)

    except Exception:
        return default_bgr

    return default_bgr


def require_cv_stack(feature: str) -> None:
    if cv2 is None or np is None or not (mp and MP_HAS_SOLUTIONS):
        raise RuntimeError(f"{feature} requires OpenCV, NumPy, and MediaPipe to be installed")


def open_video(path: str):
    cap = cv2.VideoCapture(path)  # type: ignore[arg-type]
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Could not open input video")
    return cap


def read_video_meta(cap) -> Tuple[int, int, float]:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if w <= 0 or h <= 0:
        ok, frame0 = cap.read()
        if not ok:
            raise RuntimeError("Cannot read any frame from input video")
        h, w = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return w, h, float(fps)


def make_writer(out_path: str, w: int, h: int, fps: float):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not open VideoWriter for output")
    return writer


# ---------------------------------------------------------------------------
# Video processors
# ---------------------------------------------------------------------------
def process_dots_video(input_key: str, output_key: str, dot_radius: int) -> None:
    require_cv_stack("Dots mode")

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_path = tempfile.mktemp(suffix=".mp4")

    logger.info("[dots] start radius=%s", dot_radius)

    cap = open_video(input_path)
    w, h, fps = read_video_meta(cap)
    writer = make_writer(out_path, w, h, fps)

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

                frame = cv2.resize(frame, (w, h))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)  # type: ignore[arg-type]

                canvas = np.zeros((h, w, 3), dtype=np.uint8)

                if res.pose_landmarks:
                    for lm in res.pose_landmarks.landmark:
                        if getattr(lm, "visibility", 1.0) < 0.5:
                            continue
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        if 0 <= x < w and 0 <= y < h:
                            cv2.circle(canvas, (x, y), dot_radius, (255, 255, 255), -1, lineType=cv2.LINE_8)

                writer.write(canvas)

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


def process_skeleton_video(input_key: str, output_key: str, line_bgr: Tuple[int, int, int], thickness: int) -> None:
    require_cv_stack("Skeleton mode")

    input_path = download_to_temp(input_key, suffix=".mp4")
    out_path = tempfile.mktemp(suffix=".mp4")

    logger.info("[skeleton] start color(BGR)=%s thickness=%s", line_bgr, thickness)

    cap = open_video(input_path)
    w, h, fps = read_video_meta(cap)
    writer = make_writer(out_path, w, h, fps)

    pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    connections = mp.solutions.pose.POSE_CONNECTIONS  # type: ignore[attr-defined]

    def draw_skeleton(frame_bgr, landmarks):
        hh, ww = frame_bgr.shape[:2]

        pts = []
        for lm in landmarks.landmark:
            x = int(lm.x * ww)
            y = int(lm.y * hh)
            v = getattr(lm, "visibility", 1.0)
            pts.append((x, y, v))

        # lines (sharp)
        for a, b in connections:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < 0.5 or vb < 0.5:
                continue
            if 0 <= xa < ww and 0 <= ya < hh and 0 <= xb < ww and 0 <= yb < hh:
                cv2.line(frame_bgr, (xa, ya), (xb, yb), line_bgr, thickness, lineType=cv2.LINE_8)

        # joints (sharp)
        joint_r = max(1, thickness)
        for x, y, v in pts:
            if v < 0.5:
                continue
            if 0 <= x < ww and 0 <= y < hh:
                cv2.circle(frame_bgr, (x, y), joint_r, line_bgr, -1, lineType=cv2.LINE_8)

        return frame_bgr

    detected = 0
    total = 0

    try:
        with pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                total += 1

                frame = cv2.resize(frame, (w, h))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)  # type: ignore[arg-type]

                if res.pose_landmarks:
                    detected += 1
                    frame = draw_skeleton(frame, res.pose_landmarks)

                writer.write(frame)

    finally:
        cap.release()
        writer.release()

    logger.info("[skeleton] detected frames: %d/%d", detected, total)

    # If never detected, output will look like input -> fail loudly to avoid confusion
    if total > 0 and detected == 0:
        raise RuntimeError("Skeleton not detected in any frame (video may not show full body clearly).")

    try:
        upload_from_path(out_path, output_key)
    finally:
        for p in (input_path, out_path):
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

    dot_radius = clamp_int(raw_job.get("dot_radius", 5), default=5, lo=1, hi=20)
    line_bgr = parse_color_to_bgr(raw_job.get("skeleton_line_color", "#00FF00"))
    thickness = clamp_int(raw_job.get("skeleton_line_thickness", 2), default=2, lo=1, hi=20)

    logger.info("[process_job] job_id=%s mode=%s input=%s output=%s", job_id, mode, input_key, output_key)

    job = dict(raw_job)
    job["output_key"] = output_key
    job = update_status(job, "processing")

    processing_key = f"{PROCESSING_PREFIX}/{job_id}.json"
    move_json(job_json_key, processing_key, job)

    try:
        if mode == "dots":
            process_dots_video(input_key, output_key, dot_radius=dot_radius)
        elif mode == "skeleton":
            process_skeleton_video(input_key, output_key, line_bgr=line_bgr, thickness=thickness)
        else:
            copy_video_in_s3(input_key, output_key)

        job = update_status(job, "finished", error=None)
        move_json(processing_key, f"{FINISHED_PREFIX}/{job_id}.json", job)
        logger.info("[process_job] finished job_id=%s", job_id)

    except Exception as exc:
        logger.exception("[process_job] FAILED job_id=%s", job_id)
        job = update_status(job, "failed", error=str(exc))
        move_json(processing_key, f"{FAILED_PREFIX}/{job_id}.json", job)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("====== AI People Reader Worker (No Audio) ======")
    logger.info("Using bucket: %s", AWS_BUCKET)
    logger.info("Region       : %s", AWS_REGION)
    logger.info("Poll every   : %s seconds", POLL_INTERVAL)
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
        except Exception:
            logger.exception("[main] Unexpected error")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
