# src/worker.py
"""
Background worker for AI People Reader — Johansson dots only (NO mediapipe).

Workflow:
1. Poll S3 for jobs in jobs/pending/*.json
2. Claim a job by moving JSON to jobs/processing/
3. Download input video from jobs/<job_id>/input/input.mp4
4. Render Johansson-style dot overlay using OpenCV (goodFeaturesToTrack + LK optical flow)
5. Upload output video to jobs/<job_id>/output/dots.mp4
6. Update job JSON with status/progress/message/outputs
"""

import os
import io
import json
import time
import uuid
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import boto3
import botocore
import cv2
import numpy as np

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.environ.get("S3_BUCKET")  # must be set in Render env
POLL_INTERVAL_SECONDS = int(os.environ.get("JOB_POLL_INTERVAL", "5"))

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 with timezone Z."""
    return datetime.now(timezone.utc).isoformat()


def make_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def s3_key_for_job_json(prefix: str, job_id: str) -> str:
    return f"{prefix}/{job_id}.json"


def s3_key_for_input_video(job_id: str) -> str:
    # jobs/<job_id>/input/input.mp4
    return f"{JOBS_PREFIX}/{job_id}/input/input.mp4"


def s3_key_for_output_video(job_id: str) -> str:
    # jobs/<job_id>/output/dots.mp4
    return f"{JOBS_PREFIX}/{job_id}/output/dots.mp4"


def read_json_from_s3(s3, bucket: str, key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def write_json_to_s3(s3, bucket: str, key: str, data: Dict[str, Any]):
    body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def log(msg: str):
    print(f"[worker] {datetime.utcnow().isoformat()} — {msg}", flush=True)


# ---------------------------------------------------------------------
# Job handling
# ---------------------------------------------------------------------


def find_pending_job(s3) -> Optional[str]:
    """
    Return a job_id for the first pending job, or None if none found.
    """
    resp = s3.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=PENDING_PREFIX + "/",
        MaxKeys=1,
    )
    contents = resp.get("Contents", [])
    if not contents:
        return None

    key = contents[0]["Key"]
    if not key.endswith(".json"):
        return None

    job_id = os.path.basename(key).replace(".json", "")
    return job_id


def claim_job(s3, job_id: str) -> str:
    """
    Move job JSON from pending to processing.
    Returns processing key.
    """
    pending_key = s3_key_for_job_json(PENDING_PREFIX, job_id)
    processing_key = s3_key_for_job_json(PROCESSING_PREFIX, job_id)

    log(f"Claim job {job_id}: {pending_key} -> {processing_key}")

    copy_source = {"Bucket": S3_BUCKET, "Key": pending_key}
    # copy then delete, simple "claim"
    s3.copy_object(Bucket=S3_BUCKET, CopySource=copy_source, Key=processing_key)
    s3.delete_object(Bucket=S3_BUCKET, Key=pending_key)

    return processing_key


def update_job_status(
    s3,
    job_id: str,
    processing_key: str,
    *,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    message: Optional[str] = None,
    outputs: Optional[Dict[str, Any]] = None,
):
    """
    Read job JSON, update fields, write back.
    """
    job = read_json_from_s3(s3, S3_BUCKET, processing_key)

    if status is not None:
        job["status"] = status
    if progress is not None:
        job["progress"] = progress
    if message is not None:
        job["message"] = message
    if outputs is not None:
        job["outputs"] = outputs

    job["updated_at"] = utc_now_iso()
    write_json_to_s3(s3, S3_BUCKET, processing_key, job)


# ---------------------------------------------------------------------
# Johansson dot rendering (NO mediapipe)
# ---------------------------------------------------------------------


def render_johansson_overlay(
    input_path: str,
    output_path: str,
    dot_radius: int,
    dot_count: int,
) -> None:
    """
    Render a Johansson-style point-light display from a single-person clip.

    Strategy:
    - Use goodFeaturesToTrack on grayscale frames to detect up to dot_count corners.
    - Track them across frames with calcOpticalFlowPyrLK (Lucas-Kanade).
    - Draw white circles on black background with same resolution as input.
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    if not out.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {output_path}")

    # Optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    feature_params = dict(
        maxCorners=dot_count,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )

    prev_gray = None
    p0 = None  # tracked points

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None or p0 is None or len(p0) < dot_count // 2:
            # (Re)detect features
            mask = None
            p0 = cv2.goodFeaturesToTrack(gray, mask=mask, **feature_params)
            if p0 is None:
                # No features -> output black frame
                black = np.zeros_like(frame)
                out.write(black)
                prev_gray = gray
                frame_index += 1
                continue

        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
        else:
            good_new = np.empty((0, 2), dtype=np.float32)

        # Create black frame to draw dots
        black = np.zeros_like(frame)

        for x, y in good_new:
            cv2.circle(
                black,
                (int(x), int(y)),
                dot_radius,
                (255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        out.write(black)

        prev_gray = gray.copy()
        if good_new.size > 0:
            p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = None

        frame_index += 1

    cap.release()
    out.release()


# ---------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------


def process_job(s3, job_id: str):
    """
    Download video, render Johansson dots, upload result, update job JSON.
    """
    processing_key = s3_key_for_job_json(PROCESSING_PREFIX, job_id)

    log(f"Start processing job {job_id}")
    job = read_json_from_s3(s3, S3_BUCKET, processing_key)

    # Extract parameters with defaults
    dot_radius = int(job.get("dot_radius", 5))
    dot_count = int(job.get("dot_count", 30))

    update_job_status(
        s3,
        job_id,
        processing_key,
        status="processing",
        progress=5,
        message="Downloading input video",
    )

    # Prepare temp directory
    tmp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
    input_path = os.path.join(tmp_dir, "input.mp4")
    output_path = os.path.join(tmp_dir, "dots.mp4")

    try:
        # Download input video
        input_key = s3_key_for_input_video(job_id)
        log(f"Downloading {input_key} -> {input_path}")
        s3.download_file(S3_BUCKET, input_key, input_path)

        update_job_status(
            s3,
            job_id,
            processing_key,
            status="processing",
            progress=30,
            message="Rendering Johansson dots overlay",
        )

        # Render overlay
        render_johansson_overlay(input_path, output_path, dot_radius, dot_count)

        update_job_status(
            s3,
            job_id,
            processing_key,
            status="processing",
            progress=80,
            message="Uploading output video",
        )

        # Upload result
        output_key = s3_key_for_output_video(job_id)
        log(f"Uploading {output_path} -> {output_key}")
        s3.upload_file(output_path, S3_BUCKET, output_key, ExtraArgs={"ContentType": "video/mp4"})

        # Final status
        update_job_status(
            s3,
            job_id,
            processing_key,
            status="done",
            progress=100,
            message="Completed",
            outputs={"overlay_key": output_key},
        )

        log(f"Job {job_id} completed successfully")

    except Exception as e:
        # Truncate message to avoid huge strings
        msg = repr(e)
        if len(msg) > 500:
            msg = msg[:500] + "... [truncated]"

        log(f"Job {job_id} failed: {msg}")
        update_job_status(
            s3,
            job_id,
            processing_key,
            status="failed",
            progress=100,
            message=msg,
        )

    finally:
        # Clean up temp files
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass


def main_loop():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET environment variable is not set")

    s3 = make_s3_client()
    log(f"Worker boot (Johansson DOTS ONLY, no MediaPipe)")
    log(f"AWS_REGION='{AWS_REGION}', S3_BUCKET='{S3_BUCKET}'")

    while True:
        try:
            job_id = find_pending_job(s3)
            if job_id is None:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Claim and process job
            processing_key = claim_job(s3, job_id)
            # Ensure JSON has at least minimal fields
            update_job_status(
                s3,
                job_id,
                processing_key,
                status="queued",
                progress=0,
                message="Queued",
            )

            process_job(s3, job_id)

        except botocore.exceptions.ClientError as e:
            log(f"S3 client error: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
        except Exception as e:
            log(f"Unexpected error in main loop: {repr(e)}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main_loop()
