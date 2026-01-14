# worker.py — AI People Reader Worker (Johansson dot processor)
#
# Pipeline:
#   - Poll jobs จาก jobs/pending/*.json
#   - อ่าน JSON (schema ใหม่/เก่า)
#   - โหลด input video จาก S3
#   - ประมวลผลเป็น Johansson-style dot video (พื้นดำ + จุดข้อต่อ)
#   - อัปโหลด result.mp4 ไปยัง output_key ใน S3
#   - ย้าย JSON: pending -> processing -> finished/failed

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
import io
import tempfile

import boto3
from botocore.exceptions import ClientError

import cv2
import numpy as np
import mediapipe as mp

# ----------------------------------------------------------
# Config / S3 client
# ----------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
JOB_POLL_INTERVAL = float(os.environ.get("JOB_POLL_INTERVAL", "10"))  # seconds

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET / S3_BUCKET environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

print("====== AI People Reader Worker (Johansson) ======", flush=True)
print(f"Using bucket: {AWS_BUCKET}", flush=True)
print(f"Region     : {AWS_REGION}", flush=True)
print(f"Poll every : {JOB_POLL_INTERVAL} seconds", flush=True)

# MediaPipe pose setup (สร้างครั้งเดียวใช้ทั้ง worker)
mp_pose = mp.solutions.pose


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def s3_get_json(key: str) -> Dict[str, Any]:
    print(f"[s3_get_json] key={key}", flush=True)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    print(f"[s3_put_json] key={key} size={len(body)} bytes", flush=True)
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def list_pending_objects() -> List[str]:
    """
    คืน list ของ key ทั้งหมดใต้ jobs/pending/
    """
    keys: List[str] = []
    continuation_token: Optional[str] = None

    while True:
        if continuation_token:
            resp = s3.list_objects_v2(
                Bucket=AWS_BUCKET,
                Prefix=JOBS_PENDING_PREFIX,
                ContinuationToken=continuation_token,
            )
        else:
            resp = s3.list_objects_v2(
                Bucket=AWS_BUCKET,
                Prefix=JOBS_PENDING_PREFIX,
            )

        contents = resp.get("Contents", [])
        for obj in contents:
            keys.append(obj["Key"])

        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break

    return keys


def find_one_pending_job_key() -> Optional[str]:
    """
    หา job .json ตัวแรกใน jobs/pending/
    - เลือกเฉพาะ key ที่ลงท้ายด้วย ".json"
    """
    print(f"[find_one_pending_job_key] prefix={JOBS_PENDING_PREFIX}", flush=True)
    all_keys = list_pending_objects()
    json_keys = sorted(k for k in all_keys if k.endswith(".json"))

    if not json_keys:
        print("[find_one_pending_job_key] no pending job JSON", flush=True)
        return None

    key = json_keys[0]
    print(f"[find_one_pending_job_key] found {key}", flush=True)
    return key


# ----------------------------------------------------------
# Johansson dot processing
# ----------------------------------------------------------

def process_video_to_johansson(input_key: str, output_key: str) -> None:
    """
    ดาวน์โหลด input video จาก S3 -> ประมวลผลเป็น Johansson dots -> อัปโหลด output video กลับ S3
    """
    print(f"[johansson] start processing {input_key} -> {output_key}", flush=True)

    # 1) Download input video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in_path = tmp_in.name
        print(f"[johansson] downloading to {tmp_in_path}", flush=True)
        s3.download_fileobj(AWS_BUCKET, input_key, tmp_in)

    # 2) Prepare temp output path
    tmp_out_fd, tmp_out_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_out_fd)
    print(f"[johansson] will write output to {tmp_out_path}", flush=True)

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

    print(f"[johansson] video info: {width}x{height} @ {fps} fps", flush=True)

    # Pose estimator
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[johansson] processed {frame_idx} frames", flush=True)

            # Run pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            # พื้นดำล้วน
            black = np.zeros_like(frame)

            if results.pose_landmarks:
                h, w, _ = frame.shape
                for lm in results.pose_landmarks.landmark:
                    if lm.visibility < 0.5:
                        continue
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        # จุดสีขาว ขนาด 6 px (ปรับได้)
                        cv2.circle(black, (x, y), 6, (255, 255, 255), -1)

            out.write(black)

    cap.release()
    out.release()

    # 3) Upload output video to S3
    print(f"[johansson] uploading {tmp_out_path} to s3://{AWS_BUCKET}/{output_key}", flush=True)
    with open(tmp_out_path, "rb") as f:
        s3.upload_fileobj(f, AWS_BUCKET, output_key)

    # 4) Cleanup temp files
    try:
        os.remove(tmp_in_path)
    except OSError:
        pass

    try:
        os.remove(tmp_out_path)
    except OSError:
        pass

    print("[johansson] done", flush=True)


# ----------------------------------------------------------
# Core job processing
# ----------------------------------------------------------

def process_job(pending_key: str) -> None:
    """
    ประมวลผล job หนึ่งตัวจาก jobs/pending/<job_id>.json
    """

    print(f"[process_job] start pending_key={pending_key}", flush=True)

    job = s3_get_json(pending_key)

    job_id = job.get("job_id")
    if not job_id:
        job_id = os.path.splitext(os.path.basename(pending_key))[0]
        job["job_id"] = job_id

    # รองรับทั้ง 2 schema:
    # v2 (ใหม่): input_key, output_key
    # v1 (เก่า): video_key, result_video_key
    input_key = job.get("input_key") or job.get("video_key")
    output_key = job.get("output_key") or job.get("result_video_key")

    if not input_key:
        raise RuntimeError("Job JSON missing 'input_key' / 'video_key'")

    # ถ้าไม่มี output_key/result_video_key -> ใช้ default
    if not output_key:
        output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"
        job["output_key"] = output_key
        print(
            f"[process_job] no output key in JSON, using default {output_key}",
            flush=True,
        )

    processing_key = pending_key.replace(JOBS_PENDING_PREFIX, JOBS_PROCESSING_PREFIX)

    # mark processing
    job["status"] = "processing"
    job["updated_at"] = utc_now_iso()
    job.setdefault("error", None)

    s3_put_json(processing_key, job)
    s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)
    print("[process_job] moved JSON pending -> processing", flush=True)

    try:
        # งานจริง: ทำ Johansson dots
        print(f"[process_job] johansson video {input_key} -> {output_key}", flush=True)
        process_video_to_johansson(input_key, output_key)

        finished_key = processing_key.replace(
            JOBS_PROCESSING_PREFIX, JOBS_FINISHED_PREFIX
        )
        now = utc_now_iso()
        job["status"] = "finished"
        job["finished_at"] = now
        job["updated_at"] = now
        job["error"] = None

        s3_put_json(finished_key, job)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)
        print("[process_job] moved JSON processing -> finished", flush=True)

    except Exception as exc:
        # ถ้า error -> failed
        print(f"[process_job] ERROR: {exc}", flush=True)

        failed_key = processing_key.replace(
            JOBS_PROCESSING_PREFIX, JOBS_FAILED_PREFIX
        )
        now = utc_now_iso()
        job["status"] = "failed"
        job["failed_at"] = now
        job["updated_at"] = now
        job["error"] = str(exc)

        s3_put_json(failed_key, job)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)
        print("[process_job] moved JSON processing -> failed", flush=True)

        # ให้ main เห็น error ด้วย
        raise


# ----------------------------------------------------------
# Main loop
# ----------------------------------------------------------

def main() -> None:
    print("[main] worker started", flush=True)
    while True:
        try:
            job_key = find_one_pending_job_key()
            if not job_key:
                time.sleep(JOB_POLL_INTERVAL)
                continue

            process_job(job_key)

        except ClientError as ce:
            print(f"[main] AWS ClientError: {ce}", flush=True)
            time.sleep(JOB_POLL_INTERVAL)

        except Exception as e:
            print(f"[main] Unexpected error: {e}", flush=True)
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main()
