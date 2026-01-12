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

# ============================================================
# ENV / CONFIG
# ============================================================
AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()
POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())

DOT_RADIUS_DEFAULT = 5
DOT_COUNT_DEFAULT = 60  # marker-like (set 40-80 in UI for Johansson)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ============================================================
# Mask (MOG2) — used for initial bbox + periodic correction
# ============================================================
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

MIN_MASK_AREA_RATIO = 0.006  # ~0.6% of pixels

# ============================================================
# Marker selection (Shi-Tomasi) inside bbox/mask
# ============================================================
GFTT_MAX_CORNERS_MULT = 6
GFTT_QUALITY = 0.01
GFTT_MIN_DIST = 10
GFTT_BLOCK_SIZE = 7

# ============================================================
# LK tracking (points)
# ============================================================
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

MIN_KEEP_RATIO = 0.60
REFILL_RATIO = 0.40

# ============================================================
# CSRT bbox tracking (person anchor)
# ============================================================
BBOX_EXPAND = 1.15
BBOX_MIN_W = 60
BBOX_MIN_H = 120

# Periodically re-init tracker from mask bbox IF mask is good
REINIT_TRACKER_EVERY = 45

# Clamp jitter (avoid stacking)
CLAMP_JITTER = 3

# If tracker fails too often, fallback to mask bbox
MAX_TRACKER_FAIL_STREAK = 20


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# S3 client (NO endpoint_url, NO proxies)
# ============================================================
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
    # safe CopySource (url-encode key, keep /)
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


# ============================================================
# CSRT tracker factory (compat)
# ============================================================
def create_csrt_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not available. Ensure opencv-contrib-python-headless is installed.")


# ============================================================
# ✅ Critical fix: bbox sanitizer for tracker.init()
# ============================================================
def safe_bbox(bbox, frame_w, frame_h):
    """
    Ensure bbox is valid float tuple (x, y, w, h) for OpenCV tracker.init
    Return None if invalid.
    """
    if bbox is None:
        return None
    try:
        x, y, w, h = bbox
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        if not np.isfinite([x, y, w, h]).all():
            return None

        if w <= 5 or h <= 5:
            return None

        if x < 0 or y < 0:
            return None

        if x + w > frame_w or y + h > frame_h:
            return None

        return (x, y, w, h)
    except Exception:
        return None


def expand_bbox_xywh(bbox_xywh, frame_w, frame_h, scale=BBOX_EXPAND):
    x, y, w, h = bbox_xywh
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * scale
    nh = h * scale
    x1 = max(0.0, cx - nw / 2.0)
    y1 = max(0.0, cy - nh / 2.0)
    x2 = min(float(frame_w), cx + nw / 2.0)
    y2 = min(float(frame_h), cy + nh / 2.0)
    return (x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))


def clamp_points_to_bbox(pts, bbox_xywh, frame_w, frame_h):
    if pts is None or len(pts) == 0:
        return pts
    x, y, w, h = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + w - 1.0, y + h - 1.0

    pts2 = pts.copy()
    pts2[:, 0] = np.clip(pts2[:, 0], x1, x2)
    pts2[:, 1] = np.clip(pts2[:, 1], y1, y2)

    if CLAMP_JITTER > 0:
        jitter = np.random.uniform(-CLAMP_JITTER, CLAMP_JITTER, size=pts2.shape).astype(np.float32)
        pts2 += jitter
        pts2[:, 0] = np.clip(pts2[:, 0], x1, x2)
        pts2[:, 1] = np.clip(pts2[:, 1], y1, y2)

    pts2[:, 0] = np.clip(pts2[:, 0], 0, frame_w - 1)
    pts2[:, 1] = np.clip(pts2[:, 1], 0, frame_h - 1)
    return pts2


def pts_in_bbox(pts, bbox_xywh):
    if pts is None or len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    x, y, w, h = bbox_xywh
    return (pts[:, 0] >= x) & (pts[:, 0] < x + w) & (pts[:, 1] >= y) & (pts[:, 1] < y + h)


def pts_in_mask(mask, pts, frame_w, frame_h):
    if pts is None or len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    xs = pts[:, 0]
    ys = pts[:, 1]
    inside = (xs >= 0) & (xs < frame_w) & (ys >= 0) & (ys < frame_h)
    xi = np.clip(xs.astype(np.int32), 0, frame_w - 1)
    yi = np.clip(ys.astype(np.int32), 0, frame_h - 1)
    in_mask = mask[yi, xi] > 0
    return inside & in_mask


# ============================================================
# Mask + bbox extraction
# ============================================================
def build_person_mask_bbox(frame_bgr, fgbg, k_open, k_close, k_dilate):
    h, w = frame_bgr.shape[:2]
    fg = fgbg.apply(frame_bgr)

    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.GaussianBlur(fg, (BLUR_K, BLUR_K), 0)
    _, fg = cv2.threshold(fg, RETHRESH, 255, cv2.THRESH_BINARY)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)

    bbox = None
    if KEEP_LARGEST_BLOB:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            biggest = 1 + int(np.argmax(areas))
            fg = np.where(labels == biggest, 255, 0).astype(np.uint8)

            x, y, bw, bh, area = stats[biggest]
            if bw >= BBOX_MIN_W and bh >= BBOX_MIN_H:
                bbox = (float(x), float(y), float(bw), float(bh))

    fg = cv2.dilate(fg, k_dilate, iterations=DILATE_ITER)

    min_area = int(MIN_MASK_AREA_RATIO * w * h)
    mask_ok = (np.count_nonzero(fg) >= min_area) and (bbox is not None)
    return fg, bbox, mask_ok


def sample_markers_in_bbox(gray, mask, bbox_xywh, want_n):
    if want_n <= 0:
        return None
    x, y, w, h = bbox_xywh
    x_i = int(max(0, x)); y_i = int(max(0, y))
    w_i = int(max(1, w)); h_i = int(max(1, h))
    roi_gray = gray[y_i:y_i + h_i, x_i:x_i + w_i]
    roi_mask = mask[y_i:y_i + h_i, x_i:x_i + w_i]
    if roi_gray.size == 0:
        return None

    max_corners = max(want_n * GFTT_MAX_CORNERS_MULT, want_n)
    corners = cv2.goodFeaturesToTrack(
        roi_gray,
        maxCorners=int(max_corners),
        qualityLevel=GFTT_QUALITY,
        minDistance=GFTT_MIN_DIST,
        mask=roi_mask,
        blockSize=GFTT_BLOCK_SIZE,
        useHarrisDetector=False,
    )
    if corners is None:
        return None

    pts = corners.reshape(-1, 2).astype(np.float32)
    pts[:, 0] += x_i
    pts[:, 1] += y_i

    if len(pts) > want_n:
        idx = np.random.choice(len(pts), size=want_n, replace=False)
        pts = pts[idx]

    return pts if len(pts) > 0 else None


def sample_random_in_bbox(mask, bbox_xywh, want_n):
    x, y, w, h = bbox_xywh
    x_i = int(max(0, x)); y_i = int(max(0, y))
    w_i = int(max(1, w)); h_i = int(max(1, h))
    roi = mask[y_i:y_i + h_i, x_i:x_i + w_i]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return None
    n = min(want_n, len(xs))
    idx = np.random.choice(len(xs), size=n, replace=False)
    pts = np.stack([xs[idx] + x_i, ys[idx] + y_i], axis=1).astype(np.float32)
    return pts


# ============================================================
# Renderer: Marker dots + CSRT bbox anchor + clamp (FIXED)
# ============================================================
def render_marker_csrt(input_path: str, output_path: str, dot_radius: int, dot_count: int):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_w <= 0 or frame_h <= 0:
        cap.release()
        raise RuntimeError("Invalid video size")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Cannot open VideoWriter(mp4v)")

    fgbg = cv2.createBackgroundSubtractorMOG2(MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_OPEN, KERNEL_OPEN))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_CLOSE, KERNEL_CLOSE))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))

    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask, bbox_m, mask_ok = build_person_mask_bbox(frame, fgbg, k_open, k_close, k_dilate)

    # initial bbox
    if bbox_m is not None:
        bbox_xywh = expand_bbox_xywh(bbox_m, frame_w, frame_h)
    else:
        # fallback center bbox
        bbox_xywh = (frame_w * 0.35, frame_h * 0.15, frame_w * 0.30, frame_h * 0.60)

    # ✅ FIX: init tracker ONLY if bbox is valid
    tracker = None
    safe = safe_bbox(bbox_xywh, frame_w, frame_h)
    if safe is not None:
        tracker = create_csrt_tracker()
        tracker.init(frame, safe)
    else:
        # fallback bbox that is always valid
        bbox_xywh = (frame_w * 0.35, frame_h * 0.15, frame_w * 0.30, frame_h * 0.60)
        safe = safe_bbox(bbox_xywh, frame_w, frame_h)
        tracker = create_csrt_tracker()
        tracker.init(frame, safe)

    tracker_fail_streak = 0

    # seed points inside bbox
    pts = sample_markers_in_bbox(prev_gray, mask, bbox_xywh, dot_count) if mask_ok else None
    if pts is None:
        pts = sample_random_in_bbox(mask, bbox_xywh, dot_count) if mask_ok else None
    if pts is None:
        rng = np.random.default_rng(1234)
        pts = np.stack([
            rng.integers(int(bbox_xywh[0]), int(bbox_xywh[0] + bbox_xywh[2]), size=(dot_count,)),
            rng.integers(int(bbox_xywh[1]), int(bbox_xywh[1] + bbox_xywh[3]), size=(dot_count,))
        ], axis=1).astype(np.float32)

    p0 = pts.reshape(-1, 1, 2)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask, bbox_m, mask_ok = build_person_mask_bbox(frame, fgbg, k_open, k_close, k_dilate)

        # update tracker
        if tracker is not None:
            ok_t, tb = tracker.update(frame)
            if ok_t:
                tracker_fail_streak = 0
                bbox_t = (float(tb[0]), float(tb[1]), float(tb[2]), float(tb[3]))
                safe_t = safe_bbox(bbox_t, frame_w, frame_h)
                if safe_t is not None and safe_t[2] >= BBOX_MIN_W and safe_t[3] >= BBOX_MIN_H:
                    bbox_xywh = safe_t
            else:
                tracker_fail_streak += 1

        # periodic correction: if mask ok, re-init tracker safely
        if mask_ok and bbox_m is not None and (frame_idx % REINIT_TRACKER_EVERY == 0):
            candidate = expand_bbox_xywh(bbox_m, frame_w, frame_h)
            safe_c = safe_bbox(candidate, frame_w, frame_h)
            if safe_c is not None:
                bbox_xywh = safe_c
                tracker = create_csrt_tracker()
                tracker.init(frame, safe_c)
                tracker_fail_streak = 0

        # if tracker failing too long, try reinit from mask if possible
        if tracker_fail_streak >= MAX_TRACKER_FAIL_STREAK:
            if mask_ok and bbox_m is not None:
                candidate = expand_bbox_xywh(bbox_m, frame_w, frame_h)
                safe_c = safe_bbox(candidate, frame_w, frame_h)
                if safe_c is not None:
                    bbox_xywh = safe_c
                    tracker = create_csrt_tracker()
                    tracker.init(frame, safe_c)
                    tracker_fail_streak = 0
            else:
                tracker_fail_streak = 0  # avoid infinite loop

        # Track points with LK
        p1, stt, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=LK_WIN,
            maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA
        )
        stt = stt.reshape(-1)
        p1 = p1.reshape(-1, 2)

        good = (stt == 1)

        # Always gate points to bbox (strong anti-drift)
        keep = good & pts_in_bbox(p1, bbox_xywh)
        # If mask OK, also gate to mask for extra safety
        if mask_ok:
            keep = keep & pts_in_mask(mask, p1, frame_w, frame_h)

        kept_pts = p1[keep]

        # clamp to bbox (prevents drift)
        kept_pts = clamp_points_to_bbox(kept_pts, bbox_xywh, frame_w, frame_h)

        kept_n = len(kept_pts)
        missing = dot_count - kept_n
        refill_n = 0
        if missing > 0:
            if kept_n < int(dot_count * MIN_KEEP_RATIO):
                refill_n = int(dot_count * REFILL_RATIO) + missing
            else:
                refill_n = missing

        if refill_n > 0:
            new_pts = None
            if mask_ok:
                new_pts = sample_markers_in_bbox(gray, mask, bbox_xywh, refill_n)
                if new_pts is None:
                    new_pts = sample_random_in_bbox(mask, bbox_xywh, refill_n)

            if new_pts is None:
                rng = np.random.default_rng(1000 + frame_idx)
                new_pts = np.stack([
                    rng.integers(int(bbox_xywh[0]), int(bbox_xywh[0] + bbox_xywh[2]), size=(refill_n,)),
                    rng.integers(int(bbox_xywh[1]), int(bbox_xywh[1] + bbox_xywh[3]), size=(refill_n,))
                ], axis=1).astype(np.float32)

            kept_pts = np.vstack([kept_pts, new_pts])

        if len(kept_pts) > dot_count:
            idx = np.random.choice(len(kept_pts), size=dot_count, replace=False)
            kept_pts = kept_pts[idx]

        # draw
        canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        for x, y in kept_pts:
            cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1)
        out.write(canvas)

        prev_gray = gray
        p0 = kept_pts.reshape(-1, 1, 2)

    cap.release()
    out.release()


# ============================================================
# Process one job ticket
# ============================================================
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

        update_status(job_id, "processing", 35, f"Rendering MARKER dots (CSRT anchor, SAFE bbox) dot={dot_radius}, count={dot_count} ...")
        render_marker_csrt(in_path, out_path, dot_radius=dot_radius, dot_count=dot_count)

        update_status(job_id, "processing", 90, "Uploading dots.mp4 ...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})


def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Johansson MARKER — CSRT anchor + SAFE bbox init FIXED)")
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
