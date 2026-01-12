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
DOT_COUNT_DEFAULT = 200  # ✅ default ให้เป็นรูปคนชัดขึ้น (คุณยังปรับได้จาก UI)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ----------------------------
# Motion mask tuning (สำคัญ)
# ----------------------------
MOG2_HISTORY = 250
MOG2_VAR_THRESHOLD = 25         # ✅ สูงขึ้น = ลด noise
MOG2_DETECT_SHADOWS = False

# Morphology: close หนาขึ้นช่วยเติมรู silhouette
KERNEL_OPEN = 3
KERNEL_CLOSE = 13               # ✅ จาก 9 -> 13 (คนแน่นขึ้น)

# Blur + re-threshold helps fill holes
BLUR_K = 7                      # ✅ Gaussian blur kernel
RETHRESH = 128                  # ✅ หลัง blur ให้ตัดใหม่

# Optional: keep only largest connected component (ลดฉากหลัง)
KEEP_LARGEST_BLOB = True

# ----------------------------
# Logging helpers
# ----------------------------
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

def exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False

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
    """
    copy -> delete (safe CopySource format)
    """
    encoded_key = quote(src_key, safe="/")
    copy_source = f"{S3_BUCKET}/{encoded_key}"
    s3().copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=dst_key)
    s3().delete_object(Bucket=S3_BUCKET, Key=src_key)

# ----------------------------
# Status JSON (for UI)
# ----------------------------
def update_status(job_id: str, status: str, progress: int, message: str = "", outputs=None):
    payload = {
        "job_id": job_id,
        "status": status,  # queued | processing | done | failed
        "progress": int(progress),
        "message": message,
        "updated_at": now_iso(),
        "outputs": outputs or {},
    }
    put_json(f"jobs/{job_id}/status.json", payload)

# ----------------------------
# Silhouette dots (Motion Mask via MOG2)
# ----------------------------
def render_motionmask_dots(input_path: str, output_path: str, dot_radius: int, dot_count: int):
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

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        fg = fgbg.apply(frame)  # 0..255

        # threshold ให้เป็น binary
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # ✅ blur + re-threshold เติมรูให้ silhouette หนาขึ้น
        fg = cv2.GaussianBlur(fg, (BLUR_K, BLUR_K), 0)
        _, fg = cv2.threshold(fg, RETHRESH, 255, cv2.THRESH_BINARY)

        # morphology clean
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)

        # ✅ Keep largest blob (ลด noise)
        if KEEP_LARGEST_BLOB:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            if num_labels > 1:
                # stats[0] is background
                areas = stats[1:, cv2.CC_STAT_AREA]
                biggest = 1 + int(np.argmax(areas))
                fg = np.where(labels == biggest, 255, 0).astype(np.uint8)

        ys, xs = np.where(fg > 0)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if len(xs) > 0:
            n = min(dot_count, len(xs))
            idx = np.random.choice(len(xs), size=n, replace=False)

            for x, y in zip(xs[idx], ys[idx]):
                cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1)

        out.write(canvas)

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

    # respect UI params, but provide strong defaults
    dot_radius = int(job.get("dot_radius", DOT_RADIUS_DEFAULT))
    dot_count = int(job.get("dot_count", DOT_COUNT_DEFAULT))

    output_key = f"jobs/{job_id}/output/dots.mp4"

    update_status(job_id, "processing", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dots.mp4")

        log(f"⬇️ download {input_key}")
        s3().download_file(S3_BUCKET, input_key, in_path)

        update_status(job_id, "processing", 35, f"Rendering silhouette dots (mask, dot={dot_radius}, count={dot_count}, no audio)...")
        render_motionmask_dots(in_path, out_path, dot_radius=dot_radius, dot_count=dot_count)

        update_status(job_id, "processing", 90, "Uploading dots.mp4 ...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})

def mark_ticket(processing_key: str, job_id: str, target_prefix: str):
    target_key = f"{target_prefix}{job_id}.json"
    try:
        s3_move(processing_key, target_key)
        log(f"📦 Ticket moved: {processing_key} -> {target_key}")
    except ClientError as e:
        log(f"⚠️ Ticket move skipped: {e}")

def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Silhouette DOTS — Motion Mask, tuned, NO MediaPipe)")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
            # Resume processing first
            processing_keys = list_keys(PROCESSING_PREFIX, max_keys=5)
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
            pending_keys = list_keys(PENDING_PREFIX, max_keys=5)
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
