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
DOT_COUNT_DEFAULT = 60  # marker-like density

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ----------------------------
# Motion mask tuning (used mainly for initial bbox + occasional correction)
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

# If mask too small, treat it as unreliable (use CSRT bbox)
MIN_MASK_AREA_RATIO = 0.006  # ~0.6% of pixels

# ----------------------------
# Marker selection (Shi-Tomasi)
# ----------------------------
GFTT_MAX_CORNERS_MULT = 6
GFTT_QUALITY = 0.01
GFTT_MIN_DIST = 10
GFTT_BLOCK_SIZE = 7

# ----------------------------
# LK tracking (points)
# ----------------------------
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)

# Refill behaviour
MIN_KEEP_RATIO = 0.60
REFILL_RATIO = 0.40

# ----------------------------
# CSRT bbox tracking (person anchor)
# ----------------------------
BBOX_EXPAND = 1.15        # expand bbox a bit to keep limbs inside
BBOX_MIN_W = 60
BBOX_MIN_H = 120
REINIT_TRACKER_EVERY = 45 # frames; re-init CSRT from mask bbox when mask is good

# clamp jitter (pixels)
CLAMP_JITTER = 3

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
# OpenCV CSRT tracker factory (compat)
# ----------------------------
def create_csrt_tracker():
    # OpenCV versions differ: cv2.TrackerCSRT_create or cv2.legacy.TrackerCSRT_create
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker not available. Ensure opencv-contrib-python(-headless) is installed.")

# ----------------------------
# Mask + bbox + centroid
# ----------------------------
def build_person_mask_bbox(frame_bgr, fgbg, k_open, k_close, k_dilate):
    h, w = frame_bgr.shape[:2]
    fg = fgbg.apply(frame_bgr)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

    fg = cv2.GaussianBlur(fg, (BLUR_K, BLUR_K), 0)
    _, fg = cv2.threshold(fg, RETHRESH, 255, cv2.THRESH_BINARY)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)

    bbox = None
    centroid = None

    if KEEP_LARGEST_BLOB:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            biggest = 1 + int(np.argmax(areas))
            fg = np.where(labels == biggest, 255, 0).astype(np.uint8)

            x, y, bw, bh, area = stats[biggest]
            bbox = (int(x), int(y), int(bw), int(bh))
            cx, cy = centroids[biggest]
            centroid = (float(cx), float(cy))

    fg = cv2.dilate(fg, k_dilate, iterations=DILATE_ITER)

    # Validate mask size
    min_area = int(MIN_MASK_AREA_RATIO * w * h)
    mask_ok = (np.count_nonzero(fg) >= min_area) and (bbox is not None) and (bbox[2] >= BBOX_MIN_W) and (bbox[3] >= BBOX_MIN_H)
    return fg, bbox, centroid, mask_ok

def expand_bbox(bbox, w, h, scale=BBOX_EXPAND):
    x, y, bw, bh = bbox
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    nbw = bw * scale
    nbh = bh * scale
    x1 = int(max(0, cx - nbw/2))
    y1 = int(max(0, cy - nbh/2))
    x2 = int(min(w, cx + nbw/2))
    y2 = int(min(h, cy + nbh/2))
    return (x1, y1, max(1, x2-x1), max(1, y2-y1))

def clamp_points_to_bbox(pts, bbox, w, h):
    """pts (N,2) float32; clamp to bbox and image. Add tiny jitter to avoid stacking."""
    if pts is None or len(pts) == 0:
        return pts
    x, y, bw, bh = bbox
    x1, y1, x2, y2 = x, y, x + bw - 1, y + bh - 1
    pts2 = pts.copy()
    pts2[:, 0] = np.clip(pts2[:, 0], x1, x2)
    pts2[:, 1] = np.clip(pts2[:, 1], y1, y2)
    if CLAMP_JITTER > 0:
        jitter = np.random.uniform(-CLAMP_JITTER, CLAMP_JITTER, size=pts2.shape).astype(np.float32)
        pts2 += jitter
        pts2[:, 0] = np.clip(pts2[:, 0], x1, x2)
        pts2[:, 1] = np.clip(pts2[:, 1], y1, y2)
    pts2[:, 0] = np.clip(pts2[:, 0], 0, w-1)
    pts2[:, 1] = np.clip(pts2[:, 1], 0, h-1)
    return pts2

def pts_in_bbox(pts, bbox):
    if pts is None or len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    x, y, bw, bh = bbox
    x2 = x + bw
    y2 = y + bh
    return (pts[:,0] >= x) & (pts[:,0] < x2) & (pts[:,1] >= y) & (pts[:,1] < y2)

def pts_in_mask(mask, pts, w, h):
    if pts is None or len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    xs = pts[:,0]
    ys = pts[:,1]
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xi = np.clip(xs.astype(np.int32), 0, w-1)
    yi = np.clip(ys.astype(np.int32), 0, h-1)
    in_mask = mask[yi, xi] > 0
    return inside & in_mask

def sample_markers_in_bbox(gray, mask, bbox, want_n):
    """GFTT inside bbox, and inside mask for safety."""
    if want_n <= 0:
        return None
    x, y, bw, bh = bbox
    roi_gray = gray[y:y+bh, x:x+bw]
    roi_mask = mask[y:y+bh, x:x+bw]
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
    pts = corners.reshape(-1,2).astype(np.float32)
    # convert to full coords
    pts[:,0] += x
    pts[:,1] += y
    if len(pts) > want_n:
        idx = np.random.choice(len(pts), size=want_n, replace=False)
        pts = pts[idx]
    return pts if len(pts) > 0 else None

def sample_random_in_bbox(mask, bbox, want_n):
    x, y, bw, bh = bbox
    roi = mask[y:y+bh, x:x+bw]
    ys, xs = np.where(roi > 0)
    if len(xs) == 0:
        return None
    n = min(want_n, len(xs))
    idx = np.random.choice(len(xs), size=n, replace=False)
    pts = np.stack([xs[idx] + x, ys[idx] + y], axis=1).astype(np.float32)
    return pts

# ----------------------------
# Renderer: Marker dots + CSRT bbox anchor + clamp
# ----------------------------
def render_marker_csrt(input_path: str, output_path: str, dot_radius: int, dot_count: int):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Invalid video size")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Cannot open VideoWriter(mp4v)")

    fgbg = cv2.createBackgroundSubtractorMOG2(MOG2_HISTORY, MOG2_VAR_THRESHOLD, MOG2_DETECT_SHADOWS)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_OPEN, KERNEL_OPEN))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_CLOSE, KERNEL_CLOSE))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_K, DILATE_K))

    # first frame
    ret, frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask, bbox_m, centroid, mask_ok = build_person_mask_bbox(frame, fgbg, k_open, k_close, k_dilate)

    # init bbox
    if mask_ok:
        bbox = expand_bbox(bbox_m, w, h)
    else:
        # fallback center bbox
        bbox = (int(w*0.35), int(h*0.15), int(w*0.30), int(h*0.60))

    # init CSRT tracker
    tracker = create_csrt_tracker()
    tracker.init(frame, tuple(map(float, bbox)))

    # init points
    pts = sample_markers_in_bbox(prev_gray, mask, bbox, dot_count)
    if pts is None:
        pts = sample_random_in_bbox(mask, bbox, dot_count)
    if pts is None:
        rng = np.random.default_rng(1234)
        pts = np.stack([
            rng.integers(bbox[0], bbox[0]+bbox[2], size=(dot_count,)),
            rng.integers(bbox[1], bbox[1]+bbox[3], size=(dot_count,))
        ], axis=1).astype(np.float32)

    p0 = pts.reshape(-1, 1, 2)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask, bbox_m, centroid, mask_ok = build_person_mask_bbox(frame, fgbg, k_open, k_close, k_dilate)

        # update tracker bbox
        ok_t, tb = tracker.update(frame)
        if ok_t:
            bbox_t = (int(tb[0]), int(tb[1]), int(tb[2]), int(tb[3]))
            # keep bbox within frame and reasonable size
            if bbox_t[2] >= BBOX_MIN_W and bbox_t[3] >= BBOX_MIN_H:
                bbox = (max(0,bbox_t[0]), max(0,bbox_t[1]),
                        min(w-bbox_t[0], bbox_t[2]), min(h-bbox_t[1], bbox_t[3]))
        # if mask is good, occasionally re-init tracker to correct drift
        if mask_ok and bbox_m is not None and (frame_idx % REINIT_TRACKER_EVERY == 0):
            bbox = expand_bbox(bbox_m, w, h)
            tracker = create_csrt_tracker()
            tracker.init(frame, tuple(map(float, bbox)))

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

        # Gate: must remain in bbox always
        keep = good & pts_in_bbox(p1, bbox)
        # If mask ok, also require in mask (stronger)
        if mask_ok:
            keep = keep & pts_in_mask(mask, p1, w, h)

        kept_pts = p1[keep]

        # clamp (prevent drift)
        kept_pts = clamp_points_to_bbox(kept_pts, bbox, w, h)

        # refill
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
                new_pts = sample_markers_in_bbox(gray, mask, bbox, refill_n)
                if new_pts is None:
                    new_pts = sample_random_in_bbox(mask, bbox, refill_n)

            if new_pts is None:
                rng = np.random.default_rng(1000 + frame_idx)
                new_pts = np.stack([
                    rng.integers(bbox[0], bbox[0] + bbox[2], size=(refill_n,)),
                    rng.integers(bbox[1], bbox[1] + bbox[3], size=(refill_n,))
                ], axis=1).astype(np.float32)

            kept_pts = np.vstack([kept_pts, new_pts])

        # if too many -> downsample
        if len(kept_pts) > dot_count:
            idx = np.random.choice(len(kept_pts), size=dot_count, replace=False)
            kept_pts = kept_pts[idx]

        # draw
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

        update_status(job_id, "processing", 35, f"Rendering MARKER dots (CSRT anchor, clamp) dot={dot_radius}, count={dot_count} ...")
        render_marker_csrt(in_path, out_path, dot_radius=dot_radius, dot_count=dot_count)

        update_status(job_id, "processing", 90, "Uploading dots.mp4 ...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})


def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Johansson MARKER MODE — CSRT bbox anchor + clamp, NO drift)")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
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
