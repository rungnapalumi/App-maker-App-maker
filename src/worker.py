import os
import json
import time
import tempfile
import traceback
from datetime import datetime, timezone
from urllib.parse import quote

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import cv2
import numpy as np

# ----------------------------
# ENV / CONFIG
# ----------------------------
AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()
POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())

DOT_RADIUS_DEFAULT = 5
DOT_COUNT_DEFAULT = 150  # suggest 150-250 for Johansson feel

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ----------------------------
# Motion mask tuning
# ----------------------------
MOG2_HISTORY = 250
MOG2_VAR_THRESHOLD = 25
MOG2_DETECT_SHADOWS = False

KERNEL_OPEN = 3
KERNEL_CLOSE = 13

BLUR_K = 7
RETHRESH = 128

KEEP_LARGEST_BLOB = True

# Expand mask slightly so limbs are included more often
DILATE_K = 9
DILATE_ITER = 1

# ----------------------------
# Dot tracking (Lucas-Kanade)
# ----------------------------
# How many new dots to add if lost (percentage of dot_count)
REFILL_RATIO = 0.35

# LK parameters (stable)
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

# If too few good points remain, re-seed more
MIN_KEEP_RATIO = 0.55  # if kept < 55% -> refill aggressively


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def s3():
    cfg = Config(proxies={}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def get_json(key: str) -> dict:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def put_json(key: str, data: dict):
    s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def list_keys(prefix: str, max_keys: int = 50):
    resp = s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=max_keys)
    items = resp.get("Contents", [])
    items.sort(key=lambda x: x.get("LastModified"))
    return [it["Key"] for it in items]

def s3_move(src_key: str, dst_key: str):
    encoded_key = quote(src_key, safe="/")
    copy_source = f"{S3_BUCKET}/{encoded_key}"
    s3().copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=dst_key)
    s3().delete_object(Bucket=S3_BUCKET, Key=src_key)

def update_status(job_id: str, status: str, progress: int, message: str = "", outputs=None):
    payload = {
        "job_id": job_id,
        "status": status,
        "progress": int(progress),
        "message": message,
        "updated_at": now_iso(),
        "outputs": outputs or {},
    }
    put_json(f"jobs/{job_id}/status.json", payload)

def mark_ticket(processing_key: str, job_id: str, target_prefix: str):
    target_key = f"{target_prefix}{job_id}.json"
    try:
        s3_move(processing_key, target_key)
        log(f"📦 Ticket moved: {processing_key} -> {target_key}")
    except ClientError as e:
        log(f"⚠️ Ticket move skipped: {e}")

# ----------------------------
# Mask builder
# ----------------------------
def build_person_mask(frame_bgr, fgbg, k_open, k_close, k_dilate):
    fg = fgbg.apply(frame_bgr)

    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    fg = cv2.GaussianBlur(fg, (BLUR_K, BLUR_K), 0)
    _, fg = cv2.threshold(fg, RETHRESH, 255, cv2.THRESH_BINARY)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)

    # Keep largest blob to reduce background noise
    if KEEP_LARGEST_BLOB:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            biggest = 1 + int(np.argmax(areas))
            fg = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # Dilate to include limbs and fill gaps
    fg = cv2.dilate(fg, k_dilate, iterations=DILATE_ITER)

    return fg  # uint8 0/255


def sample_points_from_mask(mask, n):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    n = min(n, len(xs))
    idx = np.random.choice(len(xs), size=n, replace=False)
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)  # (N,2)
    return pts


# ----------------------------
# Johansson-like stable dots
# - seed points from mask once
# - track with LK optical flow
# - remove points that leave mask
# - refill missing points from mask
# ----------------------------
def render_stable_johansson_dots(input_path: str, output_path: str, dot_radius: int, dot_count: int):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Invalid video size")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Cannot open VideoWriter(mp4v)")

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_OPEN, KERNEL_OPEN))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_CLOSE, KERNEL_CLOSE))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = build_person_mask(frame, fgbg, k_open, k_close, k_dilate)

    pts = sample_points_from_mask(mask, dot_count)
    if pts is None:
        # fallback: use center-ish random points if mask empty at start
        rng = np.random.default_rng(1234)
        pts = np.stack([
            rng.integers(w * 0.4, w * 0.6, size=(dot_count,)),
            rng.integers(h * 0.2, h * 0.8, size=(dot_count,))
        ], axis=1).astype(np.float32)

    # LK expects shape (N,1,2)
    p0 = pts.reshape(-1, 1, 2)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = build_person_mask(frame, fgbg, k_open, k_close, k_dilate)

        # Track points
        p1, stt, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=LK_WIN,
            maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA
        )

        # Filter good points
        stt = stt.reshape(-1)
        p1 = p1.reshape(-1, 2)
        p0_flat = p0.reshape(-1, 2)

        good = (stt == 1)

        # Also require inside frame and inside mask
        xs = p1[:, 0]
        ys = p1[:, 1]
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)

        xi = np.clip(xs.astype(np.int32), 0, w - 1)
        yi = np.clip(ys.astype(np.int32), 0, h - 1)
        in_mask = mask[yi, xi] > 0

        keep = good & inside & in_mask
        kept_pts = p1[keep]

        # Refill missing points if too many lost
        kept_n = len(kept_pts)
        target_n = dot_count

        # If mask is empty (rare), keep whatever we have (avoid full drop)
        if np.count_nonzero(mask) == 0 and kept_n > 0:
            refill_n = 0
        else:
            missing = target_n - kept_n
            if kept_n < int(target_n * MIN_KEEP_RATIO):
                # aggressive refill
                refill_n = int(target_n * REFILL_RATIO) + max(0, missing)
            else:
                refill_n = max(0, missing)

        if refill_n > 0:
            new_pts = sample_points_from_mask(mask, refill_n)
            if new_pts is not None:
                kept_pts = np.vstack([kept_pts, new_pts])

        # If still empty, fallback to avoid crash
        if len(kept_pts) == 0:
            # fallback random within frame center region
            rng = np.random.default_rng(1234 + frame_idx)
            kept_pts = np.stack([
                rng.integers(w * 0.4, w * 0.6, size=(dot_count,)),
                rng.integers(h * 0.2, h * 0.8, size=(dot_count,))
            ], axis=1).astype(np.float32)

        # If too many points after refill, downsample to dot_count
        if len(kept_pts) > dot_count:
            idx = np.random.choice(len(kept_pts), size=dot_count, replace=False)
            kept_pts = kept_pts[idx]

        # Draw dots
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for x, y in kept_pts:
            cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1)
        out.write(canvas)

        # Prepare next
        prev_gray = gray
        p0 = kept_pts.reshape(-1, 1, 2)

    cap.release()
    out.release()


def process_ticket(processing_key: str):
    job = get_json(processing_key)
    job_id = job.get("job_id") or processing_key.split("/")[-1].replace(".json", "")
    input_key = job.get("input_key")
    if not input_key:
        raise RuntimeError("Job JSON missing input_key")

    dot_radius = int(job.get("dot_radius", DOT_RADIUS_DEFAULT))
    dot_count = int(job.get("dot_count", DOT_COUNT_DEFAULT))

    output_key = f"jobs/{job_id}/output/dots.mp4"

    update_status(job_id, "processing", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dots.mp4")

        log(f"⬇️ download {input_key}")
        s3().download_file(S3_BUCKET, input_key, in_path)

        update_status(job_id, "processing", 35, f"Rendering Johansson-like stable dots (dot={dot_radius}, count={dot_count}, no audio)...")
        render_stable_johansson_dots(in_path, out_path, dot_radius=dot_radius, dot_count=dot_count)

        update_status(job_id, "processing", 90, "Uploading dots.mp4 ...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})


def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Johansson-like STABLE DOTS — Mask + Tracking, NO MediaPipe)")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
            # Resume processing first
            processing_keys = list_keys(PROCESSING_PREFIX, max_keys=3)
            if processing_keys:
                processing_key = processing_keys[0]
                job_id = processing_key.split("/")[-1].replace(".json", "")
                log(f"▶️ Resume processing {job_id}")

                try:
                    process_ticket(processing_key)
                    mark_ticket(processing_key, job_id, DONE_PREFIX)
                except Exception as e:
                    log(f"❌ Job failed {job_id}: {e!r}")
                    log(traceback.format_exc())
                    update_status(job_id, "failed", 100, message=repr(e))
                    mark_ticket(processing_key, job_id, FAILED_PREFIX)

                time.sleep(1)
                continue

            # Claim pending
            pending_keys = list_keys(PENDING_PREFIX, max_keys=3)
            if not pending_keys:
                log("⏳ heartbeat (no pending jobs)")
                time.sleep(POLL_SECONDS)
                continue

            pending_key = pending_keys[0]
            job_id = pending_key.split("/")[-1].replace(".json", "")
            processing_key = f"{PROCESSING_PREFIX}{job_id}.json"

            log(f"▶️ Claim job {job_id}: {pending_key} -> {processing_key}")
            s3_move(pending_key, processing_key)

            try:
                process_ticket(processing_key)
                mark_ticket(processing_key, job_id, DONE_PREFIX)
            except Exception as e:
                log(f"❌ Job failed {job_id}: {e!r}")
                log(traceback.format_exc())
                update_status(job_id, "failed", 100, message=repr(e))
                mark_ticket(processing_key, job_id, FAILED_PREFIX)

        except Exception as loop_err:
            log(f"❌ Worker loop error: {loop_err!r}")
            log(traceback.format_exc())
            time.sleep(3)


if __name__ == "__main__":
    main()
