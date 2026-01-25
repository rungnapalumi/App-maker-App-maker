# 2_Submit_Job.py — AI People Reader Submit Job (LEGACY-compatible)
# ✅ เขียนงานเหมือน app.py เป๊ะ: jobs/pending/<job_id>.json + input ไป jobs/pending/<job_id>/input/input.mp4
# ✅ ตัด report ออกก่อน เพื่อให้ dots/skeleton กลับมาทำงานเหมือนเดิม
# ✅ ปลอดภัย: ไม่แตะ app.py หน้าแรก

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st
import boto3
from botocore.exceptions import ClientError


# ----------------------------------------------------------
# Config (match app.py)
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

st.set_page_config(page_title="Submit Job (Legacy)", layout="wide")
st.title("🚀 Submit Job (Legacy-compatible)")
st.caption("หน้านี้สร้างงานแบบเดียวกับ app.py เพื่อให้ worker ตัวเดิมอ่านคิวได้ทันที (jobs/pending/*.json)")


# ----------------------------------------------------------
# Helpers (copy from app.py style)
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


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def presigned_get_url(key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    # บังคับให้ browser download
    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
    return s3.generate_presigned_url("get_object", Params=params, ExpiresIn=expires)


def create_job_legacy(file_bytes: bytes, mode: str, job_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    ✅ EXACT MATCH app.py

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
        "mode": mode,  # "dots" or "skeleton"
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


def find_job_json_key(job_id: str) -> Optional[str]:
    """หา job json ใน 4 โซน (pending/processing/finished/failed)"""
    candidates = [
        f"{JOBS_PENDING_PREFIX}{job_id}.json",
        f"{JOBS_PROCESSING_PREFIX}{job_id}.json",
        f"{JOBS_FINISHED_PREFIX}{job_id}.json",
        f"{JOBS_FAILED_PREFIX}{job_id}.json",
    ]
    for k in candidates:
        if s3_key_exists(k):
            return k
    return None


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
with st.expander("🔧 Environment (read-only)", expanded=False):
    st.write("AWS_BUCKET =", AWS_BUCKET)
    st.write("AWS_REGION =", AWS_REGION)

st.subheader("1) Create job (dots / skeleton)")

col1, col2 = st.columns([2, 1], vertical_alignment="top")

with col1:
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "m4v"], accept_multiple_files=False)
    note = st.text_input("Note (optional)", value="")

with col2:
    mode = st.selectbox("Mode", ["dots", "skeleton"], index=0)

    dot_radius = 5
    skeleton_color = "#00FF00"
    skeleton_thickness = 2

    if mode == "dots":
        dot_radius = st.slider("Dot size (radius px)", 1, 20, 5, 1)

    if mode == "skeleton":
        skeleton_color = st.color_picker("Line color", "#00FF00")
        skeleton_thickness = st.slider("Line thickness (px)", 1, 20, 2, 1)

if st.button("🚀 Submit job", disabled=(uploaded is None)):
    try:
        if uploaded is None:
            st.warning("Please upload a video first.")
            st.stop()

        job_fields: Dict[str, Any] = {}

        # ✅ match app.py field names exactly
        if mode == "dots":
            job_fields["dot_radius"] = int(dot_radius)

        if mode == "skeleton":
            job_fields["skeleton_line_color"] = str(skeleton_color)
            job_fields["skeleton_line_thickness"] = int(skeleton_thickness)

        # note (optional) - safe to add, worker จะ ignore ถ้าไม่ใช้
        if note.strip():
            job_fields["note"] = note.strip()

        job = create_job_legacy(uploaded.getvalue(), mode, job_fields=job_fields)

        st.session_state["last_job_id"] = job["job_id"]

        st.success(f"Created job ✅  {job['job_id']}")
        st.code(json.dumps(job, ensure_ascii=False, indent=2))

        st.info(
            "Worker จะเห็นคิวงานทันทีเพราะเขียนไปที่ "
            f"`{JOBS_PENDING_PREFIX}<job_id>.json` เหมือนหน้า app.py"
        )

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)

st.divider()
st.subheader("2) Verify job + Download output (read-only)")

job_id_check = st.text_input("Job ID to check", value=st.session_state.get("last_job_id", ""))

colA, colB = st.columns([1, 1], vertical_alignment="top")

with colA:
    if st.button("Check job.json location"):
        if not job_id_check.strip():
            st.warning("Please enter job_id")
        else:
            jid = job_id_check.strip()
            key = find_job_json_key(jid)
            if not key:
                st.error("Not found in pending/processing/finished/failed")
            else:
                st.success(f"Found ✅  {key}")
                try:
                    st.json(s3_get_json(key))
                except Exception:
                    st.warning("Found key but cannot parse JSON (unexpected format)")

with colB:
    if st.button("Check output + create download link"):
        if not job_id_check.strip():
            st.warning("Please enter job_id")
        else:
            jid = job_id_check.strip()
            out_key = f"{JOBS_OUTPUT_PREFIX}{jid}/result.mp4"

            if not s3_key_exists(out_key):
                st.info("Output not ready yet: jobs/output/<job_id>/result.mp4 ยังไม่มา")
            else:
                url = presigned_get_url(out_key, expires=3600, filename=f"{jid}_result.mp4")
                st.success("Output ready ✅")
                if hasattr(st, "link_button"):
                    st.link_button("⬇️ Download result.mp4", url)
                else:
                    st.markdown(f"[⬇️ Download result.mp4]({url})")

st.caption("หมายเหตุ: หน้านี้ intentionally ไม่ยุ่ง status.json แบบใหม่ และไม่มี report เพื่อกันพัง")
