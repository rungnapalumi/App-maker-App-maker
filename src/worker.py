# src/worker.py
# ============================================================
# AI People Reader — Worker (Johansson DOTS ONLY, NO MediaPipe)
# Uses OpenCV Optical Flow to create "Johansson-style" dots:
# - black background
# - white dots (radius=5)
# - no skeleton / no lines
# - silent mp4 output (audio removed by re-encoding frames)
#
# Queue:
#   jobs/pending/<job_id>.json
# Job format:
#   {"job_id": "...", "input_key": "jobs/<job_id>/input/input.mp4", ...}
#
# Output:
#   jobs/<job_id>/output/dot_overlay.mp4
# Status:
#   jobs/<job_id>/status.json  (queued->running->done/failed)
# ============================================================

import os
import json
import time
import tempfile
import traceback
from datetime import datetime, timezone

import boto3
from botocore.config import Config

import cv2
import numpy as np

# ----------------------------
# ENV / CONFIG
# ----------------------------
AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()
POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())

DOT_RADIUS = 5
MAX_POINTS = 120              # number of tracked points
MIN_QUALITY = 0.25            # corner quality
MIN_DISTANCE = 12             # min distance between points
FLOW_WIN = (21, 21)
FLOW_MAX_LEVEL = 3
FLOW_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ----------------------------
def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# ----------------------------
# S3 client (NO proxies / NO endpoint_url)
# ----------------------------
def s3():
    cfg = Config(
        proxies={},
        retries={"max_attempts": 5, "mode": "standard"},
    )
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

def list_pending_jobs(max_keys=10):
    resp = s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=PENDING_PREFIX, MaxKeys=max_keys)
    contents = resp.get("Contents", [])
    contents.sort(key=lambda x: x.get("LastModified"))
    return [c["Key"] for c in contents if c["Key"].endswith(".json")]

def claim_job(pending_key: str) -> str | None:
    filename = pending_key.split("/")[-1]
    processing_key = f"{PROCESSING_PREFIX}{filename}"
    if s3_exists(processing_key):
        return None
    try:
        s3().copy_object(
            Bucket=S3_BUCKET,
            Key=processing_key,
            CopySource={"Bucket": S3_BUCKET, "Key": pending_key},
        )
        s3().delete_object(Bucket=S3_BUCKET, Key=pending_key)
        return processing_key
    except Exception as e:
        log(f"❌ claim_job failed: {e!r}")
        return None

# ----------------------------
# Status updates
# ----------------------------
def update_status(job_id: str, status: str, progress: int, message: str = "", outputs: dict | None = None):
    payload = {
        "job_id": job_id,
        "status": status,
        "progress": int(progress),
        "message": message,
        "updated_at": now_iso(),
        "outputs": outputs or {},
    }
    s3_put_json(f"jobs/{job_id}/status.json", payload)

# ----------------------------
# Johansson dots renderer using optical flow
# ----------------------------
def render_johansson_dots_optical_flow(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if w <= 0 or h <= 0:
        raise RuntimeError("Invalid video size")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("Cannot open VideoWriter (mp4v)")

    # Read first frame
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Empty video")

    prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # initial points (Shi-Tomasi corners)
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=MAX_POINTS,
        qualityLevel=MIN_QUALITY,
        minDistance=MIN_DISTANCE,
        blockSize=7,
    )
    if p0 is None:
        # no features -> output black video only
        black = np.zeros((h, w, 3), dtype=np.uint8)
        out.write(black)
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            out.write(black)
        cap.release()
        out.release()
        return

    # optical flow state
    frame_i = 1
    last_redetect = 0
    REDETECT_EVERY = int(fps * 2) if fps else 60  # every ~2 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # track points
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=FLOW_WIN,
            maxLevel=FLOW_MAX_LEVEL,
            criteria=FLOW_CRITERIA,
        )

        # select good points
        if p1 is not None and st is not None:
            good_new = p1[st.flatten() == 1]
            # draw dots on black canvas
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            for pt in good_new:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(canvas, (x, y), DOT_RADIUS, (255, 255, 255), -1)
            out.write(canvas)

            # update for next frame
            prev_gray = gray
            p0 = good_new.reshape(-1, 1, 2) if len(good_new) else p0

        else:
            # fallback: write black
            out.write(np.zeros((h, w, 3), dtype=np.uint8))
            prev_gray = gray

        frame_i += 1

        # periodically re-detect features to keep dots meaningful
        if frame_i - last_redetect >= REDETECT_EVERY:
            p_new = cv2.goodFeaturesToTrack(
                prev_gray,
                maxCorners=MAX_POINTS,
                qualityLevel=MIN_QUALITY,
                minDistance=MIN_DISTANCE,
                blockSize=7,
            )
            if p_new is not None:
                p0 = p_new
            last_redetect = frame_i

    cap.release()
    out.release()

# ----------------------------
def handle_job(job: dict):
    job_id = job["job_id"]
    input_key = job["input_key"]
    overlay_key = f"jobs/{job_id}/output/dot_overlay.mp4"

    update_status(job_id, "running", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dot_overlay.mp4")

        log(f"⬇️ download {input_key}")
        s3().download_file(S3_BUCKET, input_key, in_path)

        update_status(job_id, "running", 35, "Rendering Johansson dots (optical flow, dot=5, no audio)...")
        render_johansson_dots_optical_flow(in_path, out_path)

        update_status(job_id, "running", 90, "Uploading overlay (silent mp4)...")
        log(f"⬆️ upload {overlay_key}")
        s3().upload_file(out_path, S3_BUCKET, overlay_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": overlay_key})

# ----------------------------
def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Optical Flow Johansson DOTS ONLY)")
    log(f"PYTHON_VERSION = {sys.version!r}" if "sys" in globals() else "PYTHON_VERSION unknown")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET ={S3_BUCKET!r}")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
            pending = list_pending_jobs(max_keys=5)
            if not pending:
                log("⏳ heartbeat (no pending jobs)")
                time.sleep(POLL_SECONDS)
                continue

            pending_key = pending[0]
            processing_key = claim_job(pending_key)
            if not processing_key:
                time.sleep(1)
                continue

            job = s3_get_json(processing_key)
            job_id = job.get("job_id", "unknown")
            log(f"▶️ Processing job: {job_id}")

            try:
                handle_job(job)

                s3().copy_object(
                    Bucket=S3_BUCKET,
                    Key=f"{DONE_PREFIX}{job_id}.json",
                    CopySource={"Bucket": S3_BUCKET, "Key": processing_key},
                )
                s3().delete_object(Bucket=S3_BUCKET, Key=processing_key)
                log(f"✅ Job done: {job_id}")

            except Exception as e:
                log(f"❌ Job failed: {job_id} {e!r}")
                log(traceback.format_exc())
                update_status(job_id, "failed", 100, message=repr(e))

                s3().copy_object(
                    Bucket=S3_BUCKET,
                    Key=f"{FAILED_PREFIX}{job_id}.json",
                    CopySource={"Bucket": S3_BUCKET, "Key": processing_key},
                )
                s3().delete_object(Bucket=S3_BUCKET, Key=processing_key)

        except Exception as e:
            log(f"❌ Worker loop error: {e!r}")
            log(traceback.format_exc())
            time.sleep(3)

if __name__ == "__main__":
    import sys
    main()
