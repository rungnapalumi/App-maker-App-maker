# App-maker-App-maker/src/worker.py
# ============================================================
# AI People Reader — Worker (Johansson DOTS ONLY, no MediaPipe)
# - dot radius = 5
# - black background
# - no audio (we render a new mp4)
# - polls jobs/pending/*.json
# - moves pending -> processing -> done/failed
# - writes status.json and output dots.mp4 to jobs/<job_id>/output/
#
# Job ticket format (created by app.py):
#   jobs/pending/<job_id>.json
#   {
#     "job_id": "20260111_234336__fefc6b",
#     "input_key": "jobs/20260111_234336__fefc6b/input/input.mp4",
#     ...
#   }
# ============================================================

import os
import time
import json
import tempfile
import traceback
from datetime import datetime, timezone
from urllib.parse import quote

import boto3
from botocore.config import Config

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()

POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())
DOT_RADIUS = 5

# Optical flow (Farneback)
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_FLAGS = 0

DOT_COUNT = 150  # how many dots (Johansson style)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# =========================
# Logging
# =========================
def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# =========================
# S3 client (no proxies / no endpoint_url)
# =========================
def s3():
    cfg = Config(proxies={}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def s3_put_json(key: str, data: dict):
    s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def s3_get_json(key: str) -> dict:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def s3_exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False

def list_keys(prefix: str, max_keys: int = 20):
    resp = s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=max_keys)
    items = resp.get("Contents", [])
    items.sort(key=lambda x: x.get("LastModified"))
    return [it["Key"] for it in items]

def s3_move(src_key: str, dst_key: str):
    """
    Move object within the same bucket: copy -> delete
    IMPORTANT: CopySource MUST be a string "bucket/key" and key must be URL-encoded.
    """
    # URL-encode key safely (keeps /)
    encoded_key = quote(src_key, safe="/")
    copy_source = f"{S3_BUCKET}/{encoded_key}"

    s3().copy_object(
        Bucket=S3_BUCKET,
        CopySource=copy_source,
        Key=dst_key,
    )
    s3().delete_object(Bucket=S3_BUCKET, Key=src_key)

# =========================
# Status file for Web
# =========================
def update_status(job_id: str, status: str, progress: int, message: str = "", outputs: dict | None = None):
    payload = {
        "job_id": job_id,
        "status": status,           # queued | processing | done | failed
        "progress": int(progress),
        "message": message,
        "updated_at": now_iso(),
        "outputs": outputs or {},
    }
    s3_put_json(f"jobs/{job_id}/status.json", payload)

# =========================
# Johansson dots rendering using Farneback flow
# =========================
def render_johansson_dots(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if w <= 0 or h <= 0:
        raise RuntimeError("Invalid video size")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not out.isOpened():
        raise RuntimeError("Cannot open VideoWriter(mp4v)")

    # init first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Johansson dots: start as random points
    dots = np.stack([
        np.random.randint(0, w, size=(DOT_COUNT,)),
        np.random.randint(0, h, size=(DOT_COUNT,))
    ], axis=1).astype(np.float32)  # shape (N,2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE, FLOW_ITERS,
            FLOW_POLY_N, FLOW_POLY_SIGMA, FLOW_FLAGS
        )  # shape (h,w,2)

        # advance dots by local flow vector
        xs = np.clip(dots[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(dots[:, 1].astype(np.int32), 0, h - 1)
        vecs = flow[ys, xs]  # shape (N,2)

        dots[:, 0] = np.clip(dots[:, 0] + vecs[:, 0], 0, w - 1)
        dots[:, 1] = np.clip(dots[:, 1] + vecs[:, 1], 0, h - 1)

        # draw
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for x, y in dots:
            cv2.circle(canvas, (int(x), int(y)), DOT_RADIUS, (255, 255, 255), -1)
        out.write(canvas)

        prev_gray = gray

    cap.release()
    out.release()

# =========================
# Job handler
# =========================
def process_job(job_key_processing: str):
    job = s3_get_json(job_key_processing)
    job_id = job["job_id"]
    input_key = job["input_key"]

    output_key = f"jobs/{job_id}/output/dots.mp4"

    update_status(job_id, "processing", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dots.mp4")

        log(f"⬇️ download {input_key}")
        s3().download_file(S3_BUCKET, input_key, in_path)

        update_status(job_id, "processing", 35, "Rendering Johansson dots (optical flow, dot=5, no audio)...")
        render_johansson_dots(in_path, out_path)

        update_status(job_id, "processing", 90, "Uploading output...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})

# =========================
# Main loop
# =========================
def main():
    log("✅ Worker boot (Johansson DOTS ONLY — Optical Flow)")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
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
                log(f"▶️ Processing job: {job_id}")
                process_job(processing_key)

                done_ticket = f"{DONE_PREFIX}{job_id}.json"
                log(f"✅ Move processing -> done ticket: {processing_key} -> {done_ticket}")
                s3_move(processing_key, done_ticket)

            except Exception as e:
                log(f"❌ Job failed: {job_id} {e!r}")
                log(traceback.format_exc())

                update_status(job_id, "failed", 100, message=repr(e))

                failed_ticket = f"{FAILED_PREFIX}{job_id}.json"
                log(f"❌ Move processing -> failed ticket: {processing_key} -> {failed_ticket}")
                try:
                    s3_move(processing_key, failed_ticket)
                except Exception as move_err:
                    log(f"❌ Failed to move to failed/: {move_err!r}")

        except Exception as loop_err:
            log(f"❌ Worker loop error: {loop_err!r}")
            log(traceback.format_exc())
            time.sleep(3)

if __name__ == "__main__":
    main()
