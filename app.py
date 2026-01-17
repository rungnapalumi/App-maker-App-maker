# app.py – AI People Reader Job Manager (Johansson / dots)

import os
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

# S3 paths
JOBS_PENDING_PREFIX = "jobs/pending"
JOBS_PROCESSING_PREFIX = "jobs/processing"
JOBS_FINISHED_PREFIX = "jobs/finished"
JOBS_FAILED_PREFIX = "jobs/failed"
JOBS_OUTPUT_PREFIX = "jobs/output"

# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"

def upload_bytes_to_s3(data: bytes, key: str, content_type: str):
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)

def s3_put_json(key: str, payload: dict):
    upload_bytes_to_s3(json.dumps(payload).encode("utf-8"), key, "application/json")

def s3_get_json(key: str) -> dict:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))

def list_json(prefix: str) -> List[str]:
    out = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                out.append(key)
    return out

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------

st.set_page_config(page_title="AI People Reader - Job Manager", layout="wide")

st.title("🎬 AI People Reader - Job Manager (App-maker-App-maker)")

# ==========================================================
# ① Create New Job
# ==========================================================

st.markdown("## ① Create New Job")

MODE_OPTIONS = {
    "Johansson dots – 1 person (แนะนำ)": "dots",
    "Johansson dots – 2 persons (ยังใช้ algorithm เดิม)": "dots2",
}

mode_label = st.selectbox("Mode", list(MODE_OPTIONS.keys()))
mode_value = MODE_OPTIONS[mode_label]

uploaded_file = st.file_uploader(
    "Upload video file",
    type=["mp4", "mov", "avi", "mkv"],
)

note = st.text_input("Note (optional)")

col_btn = st.columns([1, 3])[0]
if col_btn.button("Create job"):
    if not uploaded_file:
        st.error("Please upload a video file first.")
    else:
        job_id = new_job_id()

        input_key = f"{JOBS_PENDING_PREFIX}/{job_id}/input/input.mp4"
        output_key = f"{JOBS_OUTPUT_PREFIX}/{job_id}/result.mp4"
        json_key = f"{JOBS_PENDING_PREFIX}/{job_id}.json"

        # Upload video
        upload_bytes_to_s3(
            uploaded_file.read(),
            input_key,
            "video/mp4",
        )

        # Save Job JSON
        job_json = {
            "job_id": job_id,
            "status": "pending",
            "mode": mode_value,
            "input_key": input_key,
            "output_key": output_key,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "error": None,
            "note": note,
            "original_filename": uploaded_file.name,
        }

        s3_put_json(json_key, job_json)

        st.success(f"Job created successfully! Job ID: {job_id}")
        st.rerun()

# ==========================================================
# ② Job List & Download
# ==========================================================

st.markdown("## ② Job List & Download")

if st.button("🔄 Refresh job list"):
    st.rerun()

pending = list_json(JOBS_PENDING_PREFIX)
processing = list_json(JOBS_PROCESSING_PREFIX)
finished = list_json(JOBS_FINISHED_PREFIX)
failed = list_json(JOBS_FAILED_PREFIX)

rows = []

def load_jobs(keys, status_label):
    for key in keys:
        try:
            job = s3_get_json(key)
            rows.append([
                job.get("job_id"),
                status_label,
                job.get("mode"),
                job.get("created_at"),
                job.get("updated_at"),
                job.get("error"),
                job.get("note"),
                job.get("original_filename"),
            ])
        except Exception:
            pass

load_jobs(pending, "pending")
load_jobs(processing, "processing")
load_jobs(finished, "finished")
load_jobs(failed, "failed")

df = pd.DataFrame(
    rows,
    columns=["job_id", "status", "mode", "created_at", "updated_at", "error", "note", "file"],
)

st.dataframe(df, use_container_width=True)

# ==========================================================
# Download processed video
# ==========================================================

st.markdown("## ⬇️ Download processed video")
job_to_download = st.text_input("Enter job ID")

if st.button("Download"):
    if not job_to_download:
        st.error("Please enter a job ID.")
    else:
        out_key = f"{JOBS_OUTPUT_PREFIX}/{job_to_download}/result.mp4"
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=out_key)
            data = obj["Body"].read()
            st.download_button(
                label="Download processed video",
                data=data,
                file_name=f"{job_to_download}_result.mp4",
                mime="video/mp4",
            )
        except ClientError:
            st.error("Result not found or still processing.")
