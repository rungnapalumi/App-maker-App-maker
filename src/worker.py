import os
import json
import time
import shutil
import tempfile
import boto3
import cv2
import numpy as np
from datetime import datetime

# =========================
# CONFIG
# =========================
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.getenv("S3_BUCKET")

DOT_COLOR = (255, 255, 255)
FPS_FALLBACK = 25

# =========================
# S3
# =========================
s3 = boto3.client("s3", region_name=AWS_REGION)

def s3_download(key, local_path):
    s3.download_file(S3_BUCKET, key, local_path)

def s3_upload(local_path, key):
    s3.upload_file(local_path, S3_BUCKET, key)

def s3_move(src, dst):
    copy_source = {"Bucket": S3_BUCKET, "Key": src}
    s3.copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=dst)
    s3.delete_object(Bucket=S3_BUCKET, Key=src)

# =========================
# SAFE BBOX UTIL
# =========================
def safe_bbox(bbox, fw, fh, min_size=10):
    """
    bbox: (x,y,w,h) any type
    return pure python float tuple or None
    """
    try:
        x, y, w, h = bbox
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        if w < min_size or h < min_size:
            return None
        if x < 0 or y < 0:
            return None
        if x + w > fw or y + h > fh:
            return None

        return (x, y, w, h)
    except Exception:
        return None

# =========================
# TRACKER
# =========================
def create_tracker():
    return cv2.TrackerCSRT_create()

# =========================
# DOT RENDER
# =========================
def draw_dots(img, center, count, radius):
    cx, cy = center
    h, w = img.shape[:2]

    for _ in range(count):
        dx = np.random.normal(0, radius * 3)
        dy = np.random.normal(0, radius * 3)
        x = int(np.clip(cx + dx, 0, w - 1))
        y = int(np.clip(cy + dy, 0, h - 1))
        cv2.circle(img, (x, y), radius, DOT_COLOR, -1)

# =========================
# CORE PROCESS
# =========================
def render_dots(input_path, output_path, dot_radius, dot_count):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = FPS_FALLBACK

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (fw, fh),
    )

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Empty video")

    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # initial bbox = center body area
    init_bbox = (fw * 0.3, fh * 0.15, fw * 0.4, fh * 0.7)
    safe = safe_bbox(init_bbox, fw, fh)
    if safe is None:
        raise RuntimeError("Initial bbox invalid")

    tracker = create_tracker()
    tracker.init(frame, safe)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, bbox = tracker.update(frame)
        if ok:
            safe = safe_bbox(bbox, fw, fh)
            if safe is not None:
                x, y, w, h = safe
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                canvas = np.zeros_like(frame)
                draw_dots(canvas, (cx, cy), dot_count, dot_radius)
                out.write(canvas)
                continue

        # fallback: optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        y_idx, x_idx = np.where(mag > np.percentile(mag, 99))

        if len(x_idx) > 0:
            cx = int(np.mean(x_idx))
            cy = int(np.mean(y_idx))
        else:
            cx, cy = fw // 2, fh // 2

        canvas = np.zeros_like(frame)
        draw_dots(canvas, (cx, cy), dot_count, dot_radius)
        out.write(canvas)

        gray_prev = gray

    cap.release()
    out.release()

# =========================
# WORKER LOOP
# =========================
def main():
    print("✅ Worker boot (Johansson DOTS ONLY – CSRT SAFE)")

    while True:
        resp = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix="jobs/pending/",
            MaxKeys=1,
        )
        if "Contents" not in resp:
            time.sleep(3)
            continue

        job_key = resp["Contents"][0]["Key"]
        job_id = os.path.basename(job_key).replace(".json", "")
        processing_key = f"jobs/processing/{job_id}.json"

        s3_move(job_key, processing_key)

        with tempfile.TemporaryDirectory() as tmp:
            job_path = os.path.join(tmp, "job.json")
            s3_download(processing_key, job_path)

            job = json.load(open(job_path))
            dot_radius = int(job.get("dot_radius", 5))
            dot_count = int(job.get("dot_count", 30))

            input_key = f"jobs/{job_id}/input/input.mp4"
            input_path = os.path.join(tmp, "input.mp4")
            s3_download(input_key, input_path)

            output_path = os.path.join(tmp, "dots.mp4")
            render_dots(input_path, output_path, dot_radius, dot_count)

            output_key = f"jobs/{job_id}/output/dots.mp4"
            s3_upload(output_path, output_key)

            job["status"] = "done"
            job["progress"] = 100
            job["outputs"] = {"overlay_key": output_key}
            job["updated_at"] = datetime.utcnow().isoformat()

            json.dump(job, open(job_path, "w"))
            s3_upload(job_path, processing_key)

        print(f"🎉 DONE {job_id}")

if __name__ == "__main__":
    main()
