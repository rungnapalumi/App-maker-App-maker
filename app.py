# app.py — AI People Reader Job Manager (Johansson / dots / clear / skeleton)

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
# Helper functions
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
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


def create_job(
    file_bytes: bytes,
    mode: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    สร้าง job ใหม่:
      - เซฟ input video ไปที่ jobs/pending/<job_id>/input/input.mp4
      - สร้าง JSON และเซฟที่ jobs/pending/<job_id>.json

    params (optional) เช่น:
      {"dot_px": 8, "keep_audio": True}
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    # Upload video
    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    # ✅ Add params only if provided (backward compatible)
    if params:
        job["params"] = params

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)

    return job


def list_jobs() -> List[Dict[str, Any]]:
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

            job["status"] = default_status
            job["s3_key"] = key
            all_jobs.append(job)

    all_jobs.sort(key=lambda j: j.get("created_at", ""), reverse=False)
    return all_jobs


def download_output_video(job_id: str) -> bytes:
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

    # ✅ New controls (only for dots)
    dot_px = 5
    keep_audio = True

    if mode == "dots":
        st.subheader("Dots settings")

        dot_px = st.slider(
            "Dot size (px)",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="ขนาดจุด Johansson (1–20 px)",
        )

        audio_choice = st.radio(
            "Output audio",
            options=["Keep audio (มีเสียง)", "No audio (ไม่มีเสียง)"],
            index=0,
            help="ถ้าเลือก Keep audio: worker จะใช้ ffmpeg ใส่เสียงจากไฟล์ต้นฉบับกลับเข้า result.mp4",
        )
        keep_audio = (audio_choice == "Keep audio (มีเสียง)")

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

            params = None
            if mode == "dots":
                params = {
                    "dot_px": int(dot_px),
                    "keep_audio": bool(keep_audio),
                }

            job = create_job(file_bytes, mode, params=params)
            st.success(f"Created job: {job['job_id']}")
            st.json(job)

# ---------- RIGHT: Job list + download ----------
with col_right:
    st.header("Jobs")

    if st.button("Refresh job list"):
        st.rerun()

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
                    # ✅ show params if exists (nice for teacher)
                    "params": json.dumps(j.get("params", {}), ensure_ascii=False),
                }
                for j in jobs
            ]
        )
        st.dataframe(df, use_container_width=True)

        st.subheader("Download result video ↪")

        job_ids_all = [j["job_id"] for j in jobs]

        if not job_ids_all:
            st.caption("No jobs to download.")
        else:
            selected_job_id = st.selectbox(
                "Select job (will download if result.mp4 exists)",
                job_ids_all,
            )

            if st.button("Prepare download"):
                try:
                    data = download_output_video(selected_job_id)
                except ClientError as ce:
                    err_code = ce.response.get("Error", {}).get("Code")
                    if err_code == "NoSuchKey":
                        st.error(
                            "Result video for this job is not ready yet "
                            "(result.mp4 not found in S3). "
                            "Please wait a bit and refresh the job list."
                        )
                    else:
                        st.error(f"Cannot download result: {ce}")
                else:
                    st.download_button(
                        label="Download result.mp4",
                        data=data,
                        file_name=f"{selected_job_id}_result.mp4",
                        mime="video/mp4",
                    )
