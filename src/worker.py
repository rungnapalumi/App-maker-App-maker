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
DOT_COUNT_DEFAULT = 60  # ✅ Johansson marker-like density (adjustable from UI)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ----------------------------
# Motion mask (to find "person region")
# ----------------------------
MOG2_HISTORY = 250
MOG2_VAR_THRESHOLD = 25
MOG2_DETECT_SHADOWS = False

KERNEL_OPEN = 3
KERNEL_CLOSE = 13

BLUR_K = 7
RETHRESH = 128

KEEP_LARGEST_BLOB = True
DILATE_K = 11
DILATE_ITER = 1

# ----------------------------
# Marker selection (Shi-Tomasi)
# ----------------------------
# Prefer stable "important" points (corners/texture) inside person mask
GFTT_MAX_CORNERS_MULT = 6  # we detect more, then sample
GFTT_QUALITY = 0.01
GFTT_MIN_DIST = 10
GFTT_BLOCK_SIZE = 7

# ----------------------------
# Tracking (Lucas-Kanade)
# ----------------------------
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

# Refill behaviour
MIN_KEEP_RATIO = 0.60      # if < 60% points remain, refill more
REFILL_RATIO = 0.40        # refill this fraction of dot_count when low
MAX_RETRY_EMPTY_MASK = 30  # tolerate mask empty for some frames


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# S3 client (NO endpoint_url, NO proxies)
# ----------------------------
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
    fg = fgbg.apply(frame_bgr)  # 0..255
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    fg = cv2.GaussianBlur(fg, (BLUR_K, BLUR_K), 0)
    _, fg = cv2.threshold(fg, RETHRESH, 255, cv2.THRESH_BINARY)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)

    if KEEP_LARGEST_BLOB:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            biggest = 1 + int(np.argmax(areas))
            fg = np.where(labels == biggest, 255, 0).astype(np.uint8)

    fg = cv2.dilate(fg, k_dilate, iterations=DILATE_ITER)
    return fg  # uint8 0/255


def pts_in_mask(mask, pts, w, h):
    """pts shape (N,2) float32"""
    if pts is None or len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    xs = pts[:, 0]
    ys = pts[:, 1]
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xi = np.clip(xs.astype(np.int32), 0, w - 1)
    yi = np.clip(ys.astype(np.int32), 0, h - 1)
    in_mask = mask[yi, xi] > 0
    return inside & in_mask


def sample_markers_from_mask(gray, mask, want_n, w, h):
    """
    Use goodFeaturesToTrack inside the masked area (marker-like).
    Returns (N,2) float32.
    """
    if want_n <= 0:
        return None

    if np.count_nonzero(mask) == 0:
        return None

    # gftt accepts mask directly
    max_corners = max(want_n * GFTT_MAX_CORNERS_MULT, want_n)
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_corners),
        qualityLevel=GFTT_QUALITY,
        minDistance=GFTT_MIN_DIST,
        mask=mask,
        blockSize=GFTT_BLOCK_SIZE,
        useHarrisDetector=False,
    )
    if corners is None:
        return None

    pts = corners.reshape(-1, 2).astype(np.float32)

    # If too many, downsample to want_n
    if len(pts) > want_n:
        idx = np.random.choice(len(pts), size=want_n, replace=False)
        pts = pts[idx]

    # Ensure in mask
    keep = pts_in_mask(mask, pts, w, h)
    pts = pts[keep]
    return pts if len(pts) > 0 else None


# ----------------------------
# Marker-mode Johansson dots
# ----------------------------
def render_marker_johansson(input_path: str, output_path: str, dot_radius: int, dot_count: int):
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

    # First frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = build_person_mask(frame, fgbg, k_open, k_close, k_dilate)

    pts = sample_markers_from_mask(prev_gray, mask, dot_count, w, h)
    if pts is None:
        # fallback: random inside mask pixels
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            n = min(dot_count, len(xs))
            idx = np.random.choice(len(xs), size=n, replace=False)
            pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
        else:
            # final fallback
            rng = np.random.default_rng(1234)
            pts = np.stack([
                rng.integers(int(w*0.45), int(w*0.55), size=(dot_count,)),
                rng.integers(int(h*0.20), int(h*0.80), size=(dot_count,))
            ], axis=1).astype(np.float32)

    p0 = pts.reshape(-1, 1, 2)
    empty_mask_streak = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = build_person_mask(frame, fgbg, k_open, k_close, k_dilate)

        if np.count_nonzero(mask) == 0:
            empty_mask_streak += 1
        else:
            empty_mask_streak = 0

        # Track markers
        p1, stt, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=LK_WIN,
            maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA
        )

        stt = stt.reshape(-1)
        p1 = p1.reshape(-1, 2)

        good = (stt == 1)
        keep_mask = pts_in_mask(mask, p1, w, h) if empty_mask_streak < MAX_RETRY_EMPTY_MASK else good
        keep = good & keep_mask

        kept_pts = p1[keep]

        kept_n = len(kept_pts)
        target_n = dot_count

        # Decide refill amount (small refill = less blink)
        missing = target_n - kept_n
        if missing > 0:
            # if many lost, refill more
            if kept_n < int(target_n * MIN_KEEP_RATIO):
                refill_n = int(target_n * REFILL_RATIO) + missing
            else:
                refill_n = missing

            # Use marker selection first (Johansson-like)
            new_pts = sample_markers_from_mask(gray, mask, refill_n, w, h)
            if new_pts is None:
                # fallback: random in mask
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    n = min(refill_n, len(xs))
                    idx = np.random.choice(len(xs), size=n, replace=False)
                    new_pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)

            if new_pts is not None and len(new_pts) > 0:
                kept_pts = np.vstack([kept_pts, new_pts])

        # If still empty (rare), fallback
        if len(kept_pts) == 0:
            rng = np.random.default_rng(1234 + frame_idx)
            kept_pts = np.stack([
                rng.integers(int(w*0.45), int(w*0.55), size=(dot_count,)),
                rng.integers(int(h*0.20), int(h*0.80), size=(dot_count,))
            ], axis=1).astype(np.float32)

        # Downsample if too many
        if len(kept_pts) > dot_count:
            idx = np.random.choice(len(kept_pts), size=dot_count, replace=False)
            kept_pts = kept_pts[idx]

        # Draw (black background)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for x, y in kept_pts:
            cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1)
        out.write(canvas)

        prev_gray = gray
        p0 = kept_pts.reshape(-1, 1, 2)

    cap.release()
    out.release()


# ----------------------------
# Process one job ticket
# ----------------------------
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

        update_status(job_id, "processing", 35, f"Rendering MARKER dots (dot={dot_radius}, count={dot_count}, no audio)...")
        render_marker_johansson(in_path, out_path, dot_radius=dot_radius, dot_count=dot_count)

        update_status(job_id, "processing", 90, "Uploading dots.mp4 ...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})


def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Johansson MARKER MODE — GFTT + LK Tracking, NO MediaPipe)")
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
