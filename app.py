# app.py — AI People Reader Job Manager (full version)
#
# หน้าที่:
#   - ให้ผู้ใช้ upload วิดีโอ + เลือก mode (clear / dots / skeleton)
#   - สร้าง job JSON ตาม schema ที่ worker.py ใช้
#   - เซฟ input video + job JSON ลง S3
#   - แสดงรายการ jobs จากทุกสถานะ
#   - ดาวน์โหลด result.mp4 จาก jobs/output/<job_id>/result.mp4

import os
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# S3 CONFIG
# ----------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

# โครงสร้างโฟลเดอร์ใน S3
JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

# ----------------------------------------------------------
# Streamlit CONFIG
# ----------------------------------------------------------

st.set_page_config(page_title="AI People Reader - Job Manager", layout="wide")


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def utc_now_iso() -> str:
    """เวลาปัจจุบัน (UTC) ในรูปแบบ ISO string"""
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    """สร้าง job_id ใหม่ เช่น 20260114_140637__6d6c6"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    """อัปโหลด bytes ขึ้น S3"""
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    """เขียน JSON ลง S3"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_get_json(key: str) -> Dict[str, Any]:
    """อ่าน JSON จาก S3"""
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


# ----------------------------------------------------------
# Job creation
# ----------------------------------------------------------

def create_job(file_bytes: bytes, mode: str) -> Dict[str, Any]:
    """
    สร้าง job ใหม่:
      - เซฟ input video ไปที่ jobs/pending/<job_id>/input/input.mp4
      - สร้าง job JSON -> jobs/pending/<job_id>.json
    JSON schema นี้ต้อง match กับ worker.py

    Fields หลัก:
      job_id, status, mode, input_key, output_key, created_at, updated_at, error
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    # 1) Upload video
    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    # 2) Create job JSON
    now = utc_now_iso()
    job = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,          # "clear" / "dots" / "skeleton"
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)

    return job


# ----------------------------------------------------------
# Job listing
# ----------------------------------------------------------

def list_jobs() -> List[Dict[str, Any]]:
    """
    ดึง job จากทุก prefix (pending / processing / finished / failed)
    แล้วรวมเป็น list เดียว
    """
    all_jobs: List[Dict[str, Any]] = []

    prefix_status_pairs = [
        (JOBS_PENDING_PREFIX, "pending"),
        (JOBS_PROCESSING_PREFIX, "processing"),
        (JOBS_FINISHED_PREFIX, "finished"),
        (JOBS_FAILED_PREFIX, "failed"),
    ]

    for prefix, default_status in prefix_status_pairs:
        try:
            resp = s3.list_objects_v2(
                Bucket=AWS_BUCKET,
                Prefix=prefix,
            )
        except ClientError as ce:
            st.error(f"Error listing {prefix}: {ce}")
            continue

        contents = resp.get("Contents")
        if not contents:
            continue

        for obj in contents:
            key = obj["Key"]
            # ต้องเป็น *.json เท่านั้น
            if not key.endswith(".json"):
                continue

            try:
                job = s3_get_json(key)
            except ClientError as ce:
                st.warning(f"Cannot read job {key}: {ce}")
                continue

            # เผื่อ worker ยังไม่ได้เขียน status ก็ใส่ default ให้
            job.setdefault("status", default_status)
            job["s3_key"] = key
            all_jobs.append(job)

    # sort by created_at ถ้ามี (เก่าก่อน - ใหม่ทีหลัง)
    all_jobs.sort(key=lambda j: j.get("created_at", ""))
    return all_jobs


# ----------------------------------------------------------
# Download result video
# ----------------------------------------------------------

def download_output_video(job_id: str) -> bytes:
    """
    ดึง result video จาก jobs/output/<job_id>/result.mp4

    มี retry เล็กน้อย เผื่อ worker เพิ่งอัปโหลดเสร็จ
    แต่ S3 ยังไม่ propagate (eventual consistency)
    """
    key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    last_error: Exception | None = None

    for attempt in range(5):
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            return obj["Body"].read()
        except ClientError as e:
            err_code = e.response.get("Error", {}).get("Code", "")
            # ถ้ายังไม่เจอไฟล์ ให้รอแล้วลองใหม่
            if err_code in ("NoSuchKey", "404"):
                last_error = e
                time.sleep(1.0)
                continue
            # error อย่างอื่น ให้โยนออกเลย
            raise

    # ถ้า retry แล้วยังไม่เจอ
    if last_error is not None:
        raise FileNotFoundError(f"result.mp4 not found in S3 at {key}") from last_error
    else:
        raise FileNotFoundError(f"result.mp4 not found in S3 at {key}")


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------

st.title("AI People Reader - Job Manager")

col_left, col_right = st.columns([1, 2])

# ---------- LEFT: Create job ----------
with col_left:
    st.header("Create New Job")

    # ต้องให้ตรงกับ worker.py
    mode = st.selectbox(
        "Mode",
        ["clear", "dots", "skeleton"],
        index=1,  # default = "dots"
    )

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "avi"],
        accept_multiple_files=False,
    )

    if st.button("Create job"):
        if not uploaded_file:
            st.warning("Please upload a video file first.")
        else:
            file_bytes = uploaded_file.read()
            job = create_job(file_bytes, mode)
            st.success(f"Created job: {job['job_id']}")
            st.json(job)

# ---------- RIGHT: Job list ----------
with col_right:
    st.header("Jobs")

    # ปุ่ม refresh list
    if st.button("Refresh job list"):
        st.experimental_rerun()

    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet.")
    else:
        df = pd.DataFrame(
            [
                {
                    "job_id": j.get("job_id"),
                    "status": j.get("status"),
                    "mode": j.get("mode"),
                    "created_at": j.get("created_at"),
                    "updated_at": j.get("updated_at"),
                    "error": j.get("error"),
                }
                for j in jobs
            ]
        )
        st.dataframe(df, use_container_width=True)

        # Download result section
        st.subheader("Download result video")
        job_ids_finished = [j["job_id"] for j in jobs if j.get("status") == "finished"]
        if not job_ids_finished:
            st.caption("No finished jobs yet.")
        else:
            selected_job_id = st.selectbox(
                "Select finished job",
                job_ids_finished,
            )

            # ปุ่มดึงไฟล์จาก S3 แล้วแสดง download_button
            if st.button("Prepare download"):
                try:
                    data = download_output_video(selected_job_id)
                    st.success("Result video is ready. Click the button below to download.")
                    st.download_button(
                        label="Download result.mp4",
                        data=data,
                        file_name="result.mp4",
                        mime="video/mp4",
                        key=f"download_{selected_job_id}",
                    )
                except Exception as e:
                    st.error(f"Cannot download result: {e}")
