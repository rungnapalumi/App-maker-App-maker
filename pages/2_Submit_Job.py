# pages/2_Submit_Job.py — Video Analysis (วิเคราะห์วิดีโอ)
# Upload once (shared key) -> get 4 downloads:
#   1) Dots video
#   2) Skeleton video
#   3) English report (PDF + DOCX optional link)
#   4) Thai report (PDF + DOCX optional link)
#
# Uses LEGACY queue:
#   jobs/pending/<job_id>.json
# Worker must read job["input_key"] and job["output_key"] (or output_*_key for report worker)
#
# Shared input:
#   jobs/groups/<group_id>/input/input.mp4

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st
import boto3
from botocore.exceptions import ClientError


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Video Analysis (วิเคราะห์วิดีโอ)", layout="wide")

# -------------------------
# Env / S3
# -------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_OUTPUT_PREFIX = "jobs/output/"
JOBS_GROUP_PREFIX = "jobs/groups/"


# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def new_group_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}"


def guess_content_type(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".mov"):
        return "video/quicktime"
    if fn.endswith(".m4v"):
        return "video/x-m4v"
    if fn.endswith(".webm"):
        return "video/webm"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def s3_put_bytes(key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def presigned_get_url(key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        params["ResponseContentType"] = guess_content_type(filename)

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )


def create_legacy_job_json(*, job: Dict[str, Any]) -> str:
    job_id = str(job["job_id"])
    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)
    return job_json_key


# -------------------------
# UI (match your example)
# -------------------------
st.markdown("# Video Analysis (วิเคราะห์วิดีโอ)")
st.markdown("Upload your video, then click **Analysis**. (อัปโหลดวิดีโอ แล้วกด Analysis)")

uploaded = st.file_uploader(
    "Video (MP4) (วิดีโอ MP4)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
)

# keep it simple: defaults (same as your stable system)
DEFAULT_DOT_RADIUS = 5
DEFAULT_SKELETON_COLOR = "#00FF00"
DEFAULT_SKELETON_THICKNESS = 2

# optional note (kept minimal like previous)
(placeholder_col,) = st.columns(1)
note =
