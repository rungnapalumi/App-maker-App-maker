# app.py — AI People Reader Job Manager (dots + skeleton) | NO AUDIO OPTIONS

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
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def create_job(file_bytes: bytes, mode: str, job_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload input to:
      jobs/pending/<job_id>/input/input.mp4
    Save job json to:
      jobs/pending/<job_id>.json
    Output expected at:
      jobs/output/<job_id>/result.mp4
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,               # "dots" or "skeleton"
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    # top-level fields for worker
    for k, v in (job_fields or {}).items():
        job[k] = v

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)
    return job


def list_jobs() -> List[Dict[str, Any]]:
    all_jobs: List[Dict[str, Any]] = []
    pairs = [
        (JOBS_PENDING_PREFIX, "pending"),
        (JOBS_PROCESSING_PREFIX, "processing"),
        (JOBS_FINISHED_PREFIX, "finished"),
        (JOBS_FAILED_PREFIX, "failed"),
    ]

    for prefix, status in pairs:
        try:
            resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=prefix)
        except ClientError as e:
            st.error(f"List error {prefix}: {e}")
            continue

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                j = s3_get_json(key)
            except ClientError:
                continue

            j["status"] = status
            j["s3_key"] = key
            all_jobs.append(j)

    all_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=False)
    return all_jobs


def download_output_video(job_id: str) -> bytes:
    key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("AI People Reader - Job Manager (No Audio)")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Create New Job")

    mode = st.selectbox("Mode", ["dots", "skeleton"], index=0)

    dot_radius = 5
    skeleton_color = "#00FF00"
    skeleton_thickness = 2

    if mode == "dots":
        st.subheader("Dots settings")
        dot_radius = st.slider("Dot size (radius px)", 1, 20, 5, 1)

    if mode == "skeleton":
        st.subheader("Skeleton settings")
        skeleton_color = st.color_picker("Line color (default green)", "#00FF00")
        skeleton_thickness = st.slider("Line thickness (px)", 1, 20, 2, 1)

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v"],
        accept_multiple_files=False,
    )

    if st.button("Create job"):
        if not uploaded_file:
            st.warning("Please upload a video file first.")
        else:
            job_fields: Dict[str, Any] = {}

            if mode == "dots":
                job_fields["dot_radius"] = int(dot_radius)

            if mode == "skeleton":
                job_fields["skeleton_line_color"] = str(skeleton_color)
                job_fields["skeleton_line_thickness"] = int(skeleton_thickness)

            job = create_job(uploaded_file.read(), mode, job_fields=job_fields)
            st.success(f"Created job: {job['job_id']}")
            st.json(job)

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
                    "dot_radius": j.get("dot_radius"),
                    "skeleton_color": j.get("skeleton_line_color"),
                    "skeleton_thickness": j.get("skeleton_line_thickness"),
                    "created_at": j.get("created_at"),
                    "updated_at": j.get("updated_at"),
                    "error": j.get("error"),
                }
                for j in jobs
            ]
        )
        st.dataframe(df, use_container_width=True)

        st.subheader("Download result video ↪")

        job_ids_all = [j["job_id"] for j in jobs]
        selected_job_id = st.selectbox("Select job_id", job_ids_all)

        if st.button("Prepare download"):
            try:
                data = download_output_video(selected_job_id)
            except ClientError as ce:
                code = ce.response.get("Error", {}).get("Code")
                if code == "NoSuchKey":
                    st.error("Result not ready yet. Please wait and refresh.")
                else:
                    st.error(f"Download error: {ce}")
            else:
                st.download_button(
                    "Download result.mp4",
                    data=data,
                    file_name=f"{selected_job_id}_result.mp4",
                    mime="video/mp4",
                )
