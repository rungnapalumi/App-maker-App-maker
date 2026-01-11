# src/worker.py
# ============================================================
# Worker — Johansson dots (DOT ONLY, NO LINES) + Silent output
# - Polls jobs/pending/*.json
# - Downloads input video
# - Generates dot-only video on BLACK background (radius=5)
# - Writes mp4 via OpenCV (NO AUDIO)
# - Uploads overlay.mp4 + updates status.json
# ============================================================

import os
import time
import json
import traceback
import tempfile
from datetime import datetime, timezone

import boto3

AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()

POLL_SECONDS = int((os.getenv("WORKER_POLL_SECONDS") or "5").strip())
DOT_RADIUS = 5  # pixel radius as requested

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

def log(msg: str):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def s3():
    # IMPORTANT: NO endpoint_url
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def now_iso():
    return datetime.now(timezone.utc).isoformat()

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

def update_status(job_id: str, status: str, progress: int, message: str = "", outputs: dict | None = None):
    key = f"jobs/{job_id}/status.json"
    payload = {
        "job_id": job_id,
        "status": status,
        "progress": int(progress),
        "message": message,
        "updated_at": now_iso(),
        "outputs": outputs or {},
    }
    s3_put_json(key, payload)

# ------------------------------------------------------------
# Johansson DOT processing (black background, dots only)
# Output has NO AUDIO because we render frames to mp4.
# ------------------------------------------------------------
def johansson_dots_video(input_path: str, output_path: str, dot_radius: int = 5):
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if w <= 0 or h <= 0:
        raise RuntimeError("Invalid frame size")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("Cannot open VideoWriter (mp4v)")

    mp_pose = mp.solutions.pose

    # Pose model
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        frame_i = 0
        last_progress = -1

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            # BLACK background
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    if lm.visibility is not None and lm.visibility < 0.5:
                        continue
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(canvas, (x, y), dot_radius, (255, 255, 255), -1)  # white dot

            out.write(canvas)

            frame_i += 1
            if total > 0:
                prog = int((frame_i / total) * 100)
                if prog // 5 != last_progress // 5:
                    last_progress = prog
                    # caller updates status; here just return frame count
            # end loop

    cap.release()
    out.release()

def handle_job(job: dict):
    job_id = job["job_id"]
    input_key = job["input_key"]
    overlay_key = f"jobs/{job_id}/output/dot_overlay.mp4"  # explicit name

    update_status(job_id, "running", 5, "Downloading input...")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dot_overlay.mp4")

        s3().download_file(S3_BUCKET, input_key, in_path)
        update_status(job_id, "running", 25, "Generating Johansson dots (no lines)...")

        johansson_dots_video(in_path, out_path, dot_radius=DOT_RADIUS)

        update_status(job_id, "running", 90, "Uploading overlay (silent mp4)...")
        s3().upload_file(out_path, S3_BUCKET, overlay_key)

    update_status(job_id, "done", 100, "Completed", outputs={"overlay_key": overlay_key})

def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env missing")

    log("✅ Worker boot (Johansson dots)")
    log(f"AWS_REGION={AWS_REGION!r}")
    log(f"S3_BUCKET={S3_BUCKET!r}")

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

                # done marker
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
    main()
