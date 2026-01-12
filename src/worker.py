# App-maker-App-maker/src/worker.py
# ============================================================
# AI People Reader — Worker (Johansson DOTS ONLY, NO MediaPipe)
#
# ✅ Reads jobs from S3: jobs/pending/*.json
# ✅ Moves: pending -> processing -> done/failed
# ✅ Creates output: jobs/<job_id>/output/dots.mp4
# ✅ Writes status: jobs/<job_id>/status.json
# ✅ Dot radius = 5, black background, silent mp4
#
# Job ticket JSON (from app.py) should include:
#   {
#     "job_id": "20260112_031304__be1359",
#     "input_key": "jobs/20260112_031304__be1359/input/input.mp4",
#     "dot_radius": 5,
#     "dot_count": 150
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


# ----------------------------
# CONFIG / ENV
# ----------------------------
AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()
POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())

DOT_RADIUS_DEFAULT = 5
DOT_COUNT_DEFAULT = 150

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# Farneback flow params (stable defaults)
FLOW_PYR_SCALE = 0.5
FLOW_LEVELS = 3
FLOW_WINSIZE = 15
FLOW_ITERS = 3
FLOW_POLY_N = 5
FLOW_POLY_SIGMA = 1.2
FLOW_FLAGS = 0


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ----------------------------
# S3 Client (NO endpoint override, NO proxies)
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


def list_keys(prefix: str, max_keys: int = 50):
    resp = s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix, MaxKeys=max_keys)
    items = resp.get("Contents", [])
    items.sort(key=lambda x: x.get("LastModified"))
    return [it["Key"] for it in items]


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


def s3_move(src_key: str, dst_key: str):
    """
    Move object: copy -> delete
    CopySource MUST be "bucket/URL-encoded-key"
    """
    encoded_key = quote(src_key, safe="/")
    copy_source = f"{S3_BUCKET}/{encoded_key}"
    s3().copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=dst_key)
    s3().delete_object(Bucket=S3_BUCKET, Key=src_key)


def update_status(job_id: str, status: str, progress: int, message: str = "", outputs=None):
    payload = {
        "job_id": job_id,
        "status": status,           # queued | processing | done | failed
        "progress": int(progress),
        "message": message,
        "updated_at": now_iso(),
        "outputs": outputs or {},
    }
    put_json(f"jobs/{job_id}/status.json", payload)


# ----------------------------
# Johansson dots via Optical Flow (Farneback)
# Output is silent because we write a new mp4 from frames.
# ----------------------------
def render_johansson_dots(input_path: str, output_path: str, dot_radius: int, dot_count: int):
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
        raise RuntimeError("Cannot open VideoWriter (mp4v)")

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        raise RuntimeError("Cannot read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    rng = np.random.default_rng(1234)
    dots = np.stack([
        rng.integers(0, w, size=(dot_count,)),
        rng.integers(0, h, size=(dot_count,))
    ], axis=1).astype(np.float32)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            FLOW_PYR_SCALE, FLOW_LEVELS, FLOW_WINSIZE, FLOW_ITERS,
            FLOW_POLY_N, FLOW_POLY_SIGMA, FLOW_FLAGS
        )  # (h,w,2)

        xs = np.clip(dots[:, 0].astype(np.int32), 0, w - 1)
        ys = np.clip(dots[:, 1].astype(np.int32), 0, h - 1)
        vecs = flow[ys, xs]  # (N,2)

        dots[:, 0] = np.clip(dots[:, 0] + vecs[:, 0], 0, w - 1)
        dots[:, 1] = np.clip(dots[:, 1] + vecs[:, 1], 0, h - 1)

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        for x, y in dots:
            cv2.circle(canvas, (int(x), int(y)), dot_radius, (255, 255, 255), -1)

        out.write(canvas)
        prev_gray = gray

    cap.release()
    out.release()


# ----------------------------
# Process a job ticket (processing/<job_id>.json)
# ----------------------------
def process_ticket(processing_key: str):
    job = get_json(processing_key)

    job_id = job.get("job_id") or processing_key.split("/")[-1].replace(".json", "")
    input_key = job.get("input_key")
    if not input_key:
        raise RuntimeError("job missing input_key")

    dot_radius = int(job.get("dot_radius", DOT_RADIUS_DEFAULT))
    dot_count = int(job.get("dot_count", DOT_COUNT_DEFAULT))

    output_key = f"jobs/{job_id}/output/dots.mp4"

    update_status(job_id, "processing", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dots.mp4")

        log(f"⬇️ download {input_key}")
        s3().download_file(S3_BUCKET, input_key, in_path)

        update_status(job_id, "processing", 35, f"Rendering Johansson dots (optical flow, dot={dot_radius}, no audio)...")
        render_johansson_dots(in_path, out_path, dot_radius=dot_radius, dot_count=dot_count)

        update_status(job_id, "processing", 90, "Uploading output...")
        log(f"⬆️ upload {output_key}")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": output_key})


# ----------------------------
# Main loop
# ----------------------------
def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Johansson DOTS ONLY — Optical Flow, NO MediaPipe)")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
            # 1) Resume any stuck processing job first
            processing_keys = list_keys(PROCESSING_PREFIX, max_keys=5)
            if processing_keys:
                processing_key = processing_keys[0]
                job_id = processing_key.split("/")[-1].replace(".json", "")
                log(f"▶️ Resume processing: {processing_key}")

                try:
                    process_ticket(processing_key)
                    done_ticket = f"{DONE_PREFIX}{job_id}.json"
                    log(f"✅ Move processing -> done: {processing_key} -> {done_ticket}")
                    s3_move(processing_key, done_ticket)
                except Exception as e:
                    log(f"❌ Processing failed {job_id}: {e!r}")
                    log(traceback.format_exc())
                    update_status(job_id, "failed", 100, message=repr(e))
                    failed_ticket = f"{FAILED_PREFIX}{job_id}.json"
                    log(f"❌ Move processing -> failed: {processing_key} -> {failed_ticket}")
                    try:
                        s3_move(processing_key, failed_ticket)
                    except Exception as move_err:
                        log(f"❌ Move-to-failed also failed: {move_err!r}")

                time.sleep(1)
                continue

            # 2) Claim a pending job
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
                done_ticket = f"{DONE_PREFIX}{job_id}.json"
                log(f"✅ Move processing -> done: {processing_key} -> {done_ticket}")
                s3_move(processing_key, done_ticket)
            except Exception as e:
                log(f"❌ Job failed {job_id}: {e!r}")
                log(traceback.format_exc())
                update_status(job_id, "failed", 100, message=repr(e))
                failed_ticket = f"{FAILED_PREFIX}{job_id}.json"
                log(f"❌ Move processing -> failed: {processing_key} -> {failed_ticket}")
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
