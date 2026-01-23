# app.py — AI People Reader Job Manager (dots / skeleton + keep audio + dot size)
# ✅ Upload video to S3
# ✅ Create job JSON under jobs/pending/<job_id>.json (worker polls this)
# ✅ mode: dots / skeleton
# ✅ params:
#     - keep_audio: True/False
#     - dot_px: 1–20 (used for dots; harmless for skeleton)
# ✅ List jobs from pending/processing/finished/failed
# ✅ Download result video from jobs/output/<job_id>/result.mp4

import os
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
FINISHED_PREFIX = "jobs/finished/"
FAILED_PREFIX = "jobs/failed/"
OUTPUT_PREFIX = "jobs/output/"
INPUT_PREFIX = "jobs/input/"  # store raw uploads here (clean separation from pending json)

st.set_page_config(page_title="AI People Reader - Job Manager", layout="wide")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"

def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")

def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def upload_bytes_to_s3(data: bytes, key: str, content_type: str) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)

def list_json_keys(prefix: str) -> List[str]:
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs = {"Bucket": AWS_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            k = item["Key"]
            if k.endswith(".json"):
                keys.append(k)
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys

def safe_head_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except ClientError:
        return False

def download_s3_bytes(key: str) -> bytes:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()

def create_job(
    file_bytes: bytes,
    filename: str,
    mode: str,
    keep_audio: bool,
    dot_px: int,
) -> Dict[str, Any]:
    job_id = new_job_id()

    # Store original input file under jobs/input/<job_id>/input.mp4 (or preserve ext)
    ext = os.path.splitext(filename)[1].lower().strip(".")
    if ext not in ("mp4", "mov", "m4v"):
        ext = "mp4"
    input_key = f"{INPUT_PREFIX}{job_id}/input.{ext}"

    # Worker will write output here:
    output_key = f"{OUTPUT_PREFIX}{job_id}/result.mp4"

    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,  # IMPORTANT: must be "dots" or "skeleton" (exact)
        "input_key": input_key,
        "output_key": output_key,
        "params": {
            "keep_audio": bool(keep_audio),
            "dot_px": int(dot_px),
        },
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    job_json_key = f"{PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)
    return job

def load_all_jobs() -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []

    buckets = [
        (PENDING_PREFIX, "pending"),
        (PROCESSING_PREFIX, "processing"),
        (FINISHED_PREFIX, "finished"),
        (FAILED_PREFIX, "failed"),
    ]

    for prefix, status in buckets:
        try:
            keys = list_json_keys(prefix)
        except ClientError as e:
            st.error(f"Cannot list {prefix}: {e}")
            continue

        for k in keys:
            try:
                j = s3_get_json(k)
                j["status"] = status  # force status from prefix
                j["job_json_key"] = k
                jobs.append(j)
            except ClientError:
                continue

    # newest first
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return jobs

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("🎬 AI People Reader — Job Manager")

with st.expander("S3 / Environment (read-only)", expanded=False):
    st.write(f"**Bucket:** {AWS_BUCKET}")
    st.write(f"**Region:** {AWS_REGION}")
    st.caption("If jobs don’t move to processing, the worker may not be running or cannot access this bucket.")

left, right = st.columns([1, 2])

# ---------------- LEFT: Create job ----------------
with left:
    st.subheader("Create New Job")

    mode = st.selectbox(
        "Mode",
        options=["dots", "skeleton"],
        index=0,
        help="dots = Johansson dots (black background). skeleton = overlay skeleton on the original video.",
    )

    keep_audio = st.checkbox(
        "Keep audio (recommended for teaching)",
        value=True,
        help="Attach original audio back after processing.",
    )

    dot_px = st.slider(
        "Dot size (px)",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Used in dots mode. (Skeleton mode ignores this.)",
        disabled=(mode != "dots"),
    )

    up = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "m4v"],
        accept_multiple_files=False,
    )

    if st.button("🚀 Create job", use_container_width=True):
        if not up:
            st.warning("Please upload a video file first.")
        else:
            try:
                data = up.read()
                job = create_job(
                    file_bytes=data,
                    filename=up.name,
                    mode=mode,
                    keep_audio=keep_audio,
                    dot_px=dot_px,
                )
                st.success(f"Created job: {job['job_id']}")
                st.json(job)
            except ClientError as e:
                st.error(f"Create job failed: {e}")

# ---------------- RIGHT: Job list + download ----------------
with right:
    st.subheader("Jobs")

    cols = st.columns([1, 1, 1, 1])
    with cols[0]:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    jobs = load_all_jobs()

    if not jobs:
        st.info("No jobs yet.")
    else:
        df = pd.DataFrame([{
            "job_id": j.get("job_id"),
            "status": j.get("status"),
            "mode": j.get("mode"),
            "keep_audio": (j.get("params") or {}).get("keep_audio"),
            "dot_px": (j.get("params") or {}).get("dot_px"),
            "created_at": j.get("created_at"),
            "updated_at": j.get("updated_at"),
            "error": j.get("error"),
        } for j in jobs])

        st.dataframe(df, use_container_width=True, height=360)

        st.markdown("---")
        st.subheader("Download result")

        job_ids = [j.get("job_id") for j in jobs if j.get("job_id")]
        selected = st.selectbox("Select job_id", job_ids)

        result_key = f"{OUTPUT_PREFIX}{selected}/result.mp4"
        exists = safe_head_exists(result_key)

        if exists:
            if st.button("Prepare download", use_container_width=True):
                try:
                    b = download_s3_bytes(result_key)
                    st.download_button(
                        "⬇️ Download result.mp4",
                        data=b,
                        file_name=f"{selected}_result.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
                except ClientError as e:
                    st.error(f"Download failed: {e}")
        else:
            st.warning("Result not found yet (jobs/output/<job_id>/result.mp4). Wait and Refresh.")
            st.caption(f"Expected key: {result_key}")
