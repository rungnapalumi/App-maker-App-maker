# ============================================================
# AI People Reader — Worker (Johansson Dots ONLY)
# - dot only (no skeleton, no lines)
# - dot radius = 5
# - black background
# - silent mp4
# - Render-safe mediapipe import
# ============================================================

import os
import time
import json
import tempfile
import traceback
from datetime import datetime, timezone

import boto3
import cv2
import numpy as np
import mediapipe as mp   # ✅ THIS IS THE CORRECT IMPORT

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET")
DOT_RADIUS = 5
POLL_SECONDS = 5

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
DONE_PREFIX = "jobs/done/"
FAILED_PREFIX = "jobs/failed/"

# ------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

# ------------------------------------------------------------
def s3():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

# ------------------------------------------------------------
def update_status(job_id, status, progress, message="", outputs=None):
    payload = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs or {},
    }
    s3().put_object(
        Bucket=S3_BUCKET,
        Key=f"jobs/{job_id}/status.json",
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

# ------------------------------------------------------------
def johansson_dots_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            if res.pose_landmarks:
                for lm in res.pose_landmarks.landmark:
                    if lm.visibility < 0.5:
                        continue
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(canvas, (x, y), DOT_RADIUS, (255, 255, 255), -1)

            out.write(canvas)

    cap.release()
    out.release()

# ------------------------------------------------------------
def handle_job(job):
    job_id = job["job_id"]
    input_key = job["input_key"]
    output_key = f"jobs/{job_id}/output/dot_overlay.mp4"

    update_status(job_id, "running", 10, "Downloading video")

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.mp4")
        out_path = os.path.join(tmp, "dot_overlay.mp4")

        s3().download_file(S3_BUCKET, input_key, in_path)
        update_status(job_id, "running", 40, "Processing Johansson dots")

        johansson_dots_video(in_path, out_path)

        update_status(job_id, "running", 90, "Uploading result")
        s3().upload_file(out_path, S3_BUCKET, output_key)

    update_status(
        job_id,
        "done",
        100,
        "Completed",
        outputs={"overlay_key": output_key},
    )

# ------------------------------------------------------------
def main():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET not set")

    log("✅ Worker boot (Johansson dots)")
    log(f"AWS_REGION={AWS_REGION}")
    log(f"S3_BUCKET={S3_BUCKET}")

    s3().head_bucket(Bucket=S3_BUCKET)
    log("✅ S3 reachable")

    while True:
        try:
            resp = s3().list_objects_v2(Bucket=S3_BUCKET, Prefix=PENDING_PREFIX)
            items = resp.get("Contents", [])

            if not items:
                log("⏳ heartbeat (no pending jobs)")
                time.sleep(POLL_SECONDS)
                continue

            key = items[0]["Key"]
            job = json.loads(
                s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            )

            job_id = job["job_id"]
            log(f"▶️ Processing job {job_id}")

            s3().copy_object(
                Bucket=S3_BUCKET,
                Key=f"{PROCESSING_PREFIX}{job_id}.json",
                CopySource={"Bucket": S3_BUCKET, "Key": key},
            )
            s3().delete_object(Bucket=S3_BUCKET, Key=key)

            try:
                handle_job(job)
                s3().copy_object(
                    Bucket=S3_BUCKET,
                    Key=f"{DONE_PREFIX}{job_id}.json",
                    CopySource={
                        "Bucket": S3_BUCKET,
                        "Key": f"{PROCESSING_PREFIX}{job_id}.json",
                    },
                )
            except Exception as e:
                log(f"❌ Job failed: {e}")
                log(traceback.format_exc())
                update_status(job_id, "failed", 100, str(e))
            finally:
                s3().delete_object(
                    Bucket=S3_BUCKET,
                    Key=f"{PROCESSING_PREFIX}{job_id}.json",
                )

        except Exception as e:
            log(f"❌ Worker loop error: {e}")
            log(traceback.format_exc())
            time.sleep(3)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
