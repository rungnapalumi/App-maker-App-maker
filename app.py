# app.py  — AI People Reader Job Manager (Johansson dots – 1 person)
# --------------------------------------------------------------------
# หน้าที่ของ app นี้
#   1) ให้ผู้ใช้ upload วิดีโอสัมภาษณ์
#   2) สร้าง job JSON ไปที่ S3 (โฟลเดอร์ jobs/pending/)
#   3) worker.py จะอ่าน JSON + วิดีโอจาก S3 แล้วประมวลผล
#   4) app ดึงสถานะจาก jobs/(pending|processing|finished|failed)
#   5) เมื่อเสร็จแล้ว กด Download เพื่อโหลด result.mp4 ได้ (ผ่าน Streamlit)

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

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PREFIX = "jobs"
JOBS_PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
JOBS_PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
JOBS_FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
JOBS_FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
JOBS_OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"


# --------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    """สร้าง job id ใหม่ให้จำง่าย"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "video/mp4") -> None:
    s3.upload_fileobj(
        io.BytesIO(data),
        AWS_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def s3_put_json(key: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_get_json(key: str) -> Optional[dict]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    except ClientError:
        return None
    data = obj["Body"].read()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


def list_job_json_keys(prefix: str) -> List[str]:
    """คืนลิสต์ key ของ *.json ใต้ prefix ที่กำหนด"""
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                keys.append(key)
    return keys


def collect_all_jobs() -> List[Dict[str, Any]]:
    """อ่าน JSON จาก pending / processing / finished / failed มารวมเป็น list เดียว"""
    jobs: List[Dict[str, Any]] = []

    for prefix in [
        JOBS_PENDING_PREFIX,
        JOBS_PROCESSING_PREFIX,
        JOBS_FINISHED_PREFIX,
        JOBS_FAILED_PREFIX,
    ]:
        for key in list_job_json_keys(prefix):
            data = s3_get_json(key)
            if not data:
                continue

            job_id = data.get("job_id") or os.path.splitext(os.path.basename(key))[0]
            status = data.get("status", "unknown")
            mode = data.get("mode", "-")
            created_at = data.get("created_at") or data.get("created_at_utc")
            updated_at = data.get("updated_at") or data.get("updated_at_utc")
            error = data.get("error")
            note = data.get("user_note")
            filename = data.get("original_filename")

            jobs.append(
                {
                    "job_id": job_id,
                    "status": status,
                    "mode": mode,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "error": error,
                    "note": note,
                    "file": filename,
                }
            )

    # sort: newest first
    def _parse_dt(x: Any) -> float:
        if not x:
            return 0.0
        try:
            return datetime.fromisoformat(str(x)).timestamp()
        except Exception:
            return 0.0

    jobs.sort(key=lambda j: _parse_dt(j.get("created_at")), reverse=True)
    return jobs


def get_finished_job(job_id: str) -> Optional[dict]:
    """
    อ่าน JSON ของ job ที่เสร็จแล้วจาก S3 (โฟลเดอร์ finished)
    ถ้าไม่เจอหรือ error ให้คืน None
    """
    key = f"{JOBS_FINISHED_PREFIX}/{job_id}.json"
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    except Exception:
        return None

    data = obj["Body"].read()
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------

st.set_page_config(
    page_title="AI People Reader - Job Manager",
    layout="wide",
)

st.title("🎬 AI People Reader - Job Manager (App-maker-App-maker)")

left, right = st.columns([1.1, 1.5])

# --------------------------------------------------------------------
# ① Create New Job
# --------------------------------------------------------------------
with left:
    st.subheader("① Create New Job")

    # ตอนนี้ worker รองรับ mode เดียวคือ 'dots'
    mode_label = st.selectbox(
        "Mode",
        options=[
            "Johansson dots – 1 person (แนวน้ำ)",
        ],
        index=0,
    )
    mode_value = "dots"  # map label → internal mode

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "mpeg4"],
        help="Limit 1GB per file • MP4, MOV, M4V, MPEG4",
    )

    note = st.text_input("Note (optional)", value="")

    if st.button("Create job", type="primary"):
        if not uploaded_file:
            st.error("Please upload a video file first.")
        else:
            try:
                job_id = new_job_id()

                input_key = f"{JOBS_PENDING_PREFIX}/{job_id}/input/input.mp4"
                output_key = f"{JOBS_OUTPUT_PREFIX}/{job_id}/result.mp4"
                json_key = f"{JOBS_PENDING_PREFIX}/{job_id}.json"

                # 1) upload video
                upload_bytes_to_s3(uploaded_file.read(), input_key, content_type="video/mp4")

                # 2) create job JSON
                now = utc_now_iso()
                job_payload = {
                    "job_id": job_id,
                    "status": "pending",
                    "mode": mode_value,
                    "input_key": input_key,
                    "output_key": output_key,
                    "created_at": now,
                    "updated_at": now,
                    "error": None,
                    "user_note": note,
                    "original_filename": uploaded_file.name,
                }

                s3_put_json(json_key, job_payload)

                st.success(f"Job created! Job ID: {job_id}")
                st.info("ไปที่ด้านขวาเพื่อดูสถานะ และใช้ Job ID เพื่อดาวน์โหลดผลลัพธ์นะคะ ✨")

            except Exception as exc:
                st.error(f"Error creating job: {exc}")


# --------------------------------------------------------------------
# ② Job List & Download
# --------------------------------------------------------------------
with right:
    st.subheader("② Job List & Download")

    if st.button("🔁 Refresh job list"):
        # การกดปุ่มจะทำให้สคริปต์ rerun อยู่แล้ว
        pass

    jobs = collect_all_jobs()
    if jobs:
        df = pd.DataFrame(jobs)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No jobs yet. ลองสร้างงานใหม่ทางฝั่งซ้ายก่อนนะคะ")

# --------------------------------------------------------------------
# Download processed video
# --------------------------------------------------------------------

st.markdown("---")
st.header("⬇️ Download processed video")

job_id_for_download = st.text_input("Enter job ID", "")

# ตัวแปรสำหรับเก็บ bytes ของวิดีโอ (ถ้าหาเจอ)
video_bytes: Optional[bytes] = None
download_filename: str = "result.mp4"

if st.button("Download"):
    job_id_for_download = job_id_for_download.strip()

    if not job_id_for_download:
        st.error("Please enter job ID.")
    else:
        output_key: Optional[str] = None

        # 1) พยายามอ่าน JSON จาก jobs/finished/ (เวอร์ชันใหม่)
        job = get_finished_job(job_id_for_download)

        if job and job.get("status") == "finished":
            output_key = job.get("output_key")
            if not output_key:
                # กันกรณี JSON เก่าที่ไม่มี output_key
                output_key = f"{JOBS_OUTPUT_PREFIX}/{job_id_for_download}/result.mp4"

        # 2) ถ้ายังไม่มี ให้ลอง fallback หาไฟล์ตรง ๆ ตาม pattern เดิม
        if not output_key:
            fallback_key = f"{JOBS_OUTPUT_PREFIX}/{job_id_for_download}/result.mp4"
            try:
                s3.head_object(Bucket=AWS_BUCKET, Key=fallback_key)
                output_key = fallback_key
            except ClientError:
                output_key = None

        if not output_key:
            st.error(
                "Result not found or still processing.\n\n"
                "ถ้าเป็นงานเก่าก่อนปรับระบบ อาจต้องรันวิดีโอนั้นใหม่อีกครั้งนะคะ 💛"
            )
        else:
            try:
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=output_key)
                video_bytes = obj["Body"].read()
                download_filename = f"{job_id_for_download}_dots.mp4"
                st.success("Video is ready. กดปุ่มด้านล่างเพื่อดาวน์โหลดนะคะ 👇")
            except Exception as exc:
                st.error(f"Error reading video from S3: {exc}")
                video_bytes = None

# ถ้าเรามี video_bytes แล้ว ค่อยโชว์ปุ่มดาวน์โหลด
if video_bytes:
    st.download_button(
        label="Download processed video",
        data=video_bytes,
        file_name=download_filename,
        mime="video/mp4",
    )
