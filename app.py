# app.py — AI People Reader Job Manager (final version)
#
# หน้าที่:
#   - ให้ผู้ใช้ upload วิดีโอ + เลือก mode
#   - สร้าง job JSON ตาม schema เดียวกับ worker.py
#   - เซฟ input video + job JSON ลง S3
#   - แสดงรายการ jobs ทั้งหมด

import os
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

st.set_page_config(page_title="AI People Reader - Job Manager", layout="wide")


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    st.write(f"Uploading to s3://{AWS_BUCKET}/{key}")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    data = obj["Body"].read()
    return json.loads(data.decode("utf-8"))


def create_job(file_bytes: bytes, mode: str) -> Dict[str, Any]:
    """
    สร้าง job ใหม่:
      - เซฟ input video ไปที่ jobs/pending/<job_id>/input/input.mp4
      - สร้าง JSON และเซฟที่ jobs/pending/<job_id>.json
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    # Upload video
    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)

    return job


def list_jobs() -> List[Dict[str, Any]]:
    """
    ดึง job จากทุก prefix (pending/processing/finished/failed)
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
            if not key.endswith(".json"):
                continue

            try:
                job = s3_get_json(key)
            except ClientError as ce:
                st.warning(f"Cannot read job {key}: {ce}")
                continue

            job.setdefault("status", default_status)
            job["s3_key"] = key
            all_jobs.append(job)

    # sort by created_at ถ้ามี
    all_jobs.sort(key=lambda j: j.get("created_at", ""))
    return all_jobs


def download_output_video(job_id: str) -> bytes:
    """
    ดึง result video จาก jobs/output/<job_id>/result.mp4
    """
    key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------

st.title("AI People Reader - Job Manager")

col_left, col_right = st.columns([1, 2])

# ---------- LEFT: Create job ----------
with col_left:
    st.header("Create New Job")

    mode = st.selectbox("Mode", ["dots", "clear", "skeleton"], index=0)

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v"],
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

    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet.")
    else:
        # Show as table
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
            if st.button("Download result.mp4"):
                try:
                    data = download_output_video(selected_job_id)
                    st.download_button(
                        label="Download result.mp4",
                        data=data,
                        file_name="result.mp4",
                        mime="video/mp4",
                    )
                except ClientError as ce:
                    st.error(f"Cannot download result: {ce}")
