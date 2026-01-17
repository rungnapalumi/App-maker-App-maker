# app.py — AI People Reader Job Manager (App-maker-App-maker)
#
# หน้าที่:
#   - ให้ผู้ใช้ upload วิดีโอ
#   - เลือก mode ว่าจะให้ worker ทำ Johansson dots หรือ copy วิดีโอ
#   - สร้าง job JSON ไปไว้ที่ jobs/pending/<job_id>.json
#   - ให้โหลด result ที่ jobs/output/<job_id>/result.mp4
#
# NOTE: โค้ดนี้ออกแบบให้เข้าคู่กับ worker.py เวอร์ชันที่ Rung ส่งมา (process_dots_video + passthrough)

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

JOBS_PREFIX = "jobs"
PENDING_PREFIX = f"{JOBS_PREFIX}/pending"
PROCESSING_PREFIX = f"{JOBS_PREFIX}/processing"
FINISHED_PREFIX = f"{JOBS_PREFIX}/finished"
FAILED_PREFIX = f"{JOBS_PREFIX}/failed"
OUTPUT_PREFIX = f"{JOBS_PREFIX}/output"

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


def create_job(file_bytes: bytes, filename: str, mode: str, user_note: str = "") -> Dict[str, Any]:
    """
    สร้าง job ใหม่:
      - input video:  jobs/pending/<job_id>/input/input.mp4
      - output video: jobs/output/<job_id>/result.mp4
      - job json:     jobs/pending/<job_id>.json
    """
    job_id = new_job_id()

    input_key = f"{PENDING_PREFIX}/{job_id}/input/input.mp4"
    output_key = f"{OUTPUT_PREFIX}/{job_id}/result.mp4"
    job_json_key = f"{PENDING_PREFIX}/{job_id}.json"

    # upload video
    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,  # ส่งไปให้ worker ตัดสินใจ
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "user_note": user_note or "",
        "original_filename": filename,
    }

    s3_put_json(job_json_key, job)
    return job


def list_jobs() -> List[Dict[str, Any]]:
    """
    รวม job จาก pending/processing/finished/failed
    ใช้ prefix เป็นตัวบอกสถานะ (ไม่พึ่งพา field status ใน JSON อย่างเดียว)
    """
    all_jobs: List[Dict[str, Any]] = []

    prefix_status_pairs = [
        (PENDING_PREFIX, "pending"),
        (PROCESSING_PREFIX, "processing"),
        (FINISHED_PREFIX, "finished"),
        (FAILED_PREFIX, "failed"),
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

            job["status"] = default_status  # แหล่งความจริงเรื่อง status
            job["s3_key"] = key
            all_jobs.append(job)

    # sort ตามเวลาสร้าง (จากเก่าไปใหม่)
    all_jobs.sort(key=lambda j: j.get("created_at", ""), reverse=False)
    return all_jobs


def download_output_video(job_id: str) -> bytes:
    """
    ดึงไฟล์ result.mp4 จาก jobs/output/<job_id>/result.mp4
    """
    key = f"{OUTPUT_PREFIX}/{job_id}/result.mp4"
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------

st.title("🎬 AI People Reader - Job Manager (App-maker-App-maker)")

col_left, col_right = st.columns([1, 2])

# ---------- LEFT: Create job ----------
with col_left:
    st.header("① Create New Job")

    # ให้คุณครูเลือกโหมดอ่านง่ายบนจอ
    mode_label = st.selectbox(
        "Mode",
        [
            "Johansson dots – 1 person (แนะนำ)",
            "Johansson dots – 2 persons (ยังใช้ algorithm เดิม)",
            "Copy video (no processing)",
        ],
        index=0,
    )

    # map label -> mode ที่ worker เข้าใจ
    if mode_label.startswith("Johansson dots"):
        # ตอนนี้ worker มีแค่ mode="dots" ตัวเดียว
        # เราเลยให้ทั้ง single/multi ใช้ mode="dots" เหมือนกัน
        # (ต่างกันแค่ label บน UI สำหรับการสื่อสารกับคุณครู)
        mode = "dots"
    else:
        # โหมด copy วิดีโอ
        mode = "passthrough"

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "avi", "mkv", "mpeg4"],
        accept_multiple_files=False,
    )

    user_note = st.text_input("Note (optional, สำหรับ Rung/คุณครู)", "")

    if st.button("Create job", type="primary"):
        if not uploaded_file:
            st.warning("Please upload a video file first.")
        else:
            file_bytes = uploaded_file.read()
            job = create_job(
                file_bytes=file_bytes,
                filename=uploaded_file.name,
                mode=mode,
                user_note=user_note,
            )
            st.success(f"Created job: {job['job_id']}")
            with st.expander("Job JSON (frontend)", expanded=False):
                st.json(job)


# ---------- RIGHT: Job list + download ----------
with col_right:
    st.header("② Job List & Download")

    if st.button("🔄 Refresh job list"):
        st.rerun()

    jobs = list_jobs()
    if not jobs:
        st.info("ยังไม่พบงานใด ๆ ลองสร้างงานทางด้านซ้าย")
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
                    "note": j.get("user_note", ""),
                    "file": j.get("original_filename", ""),
                }
                for j in jobs
            ]
        )
        st.dataframe(df, use_container_width=True)

        st.subheader("⬇ Download processed video")

        job_ids_all = [j["job_id"] for j in jobs]
        selected_job_id = st.selectbox("Select job ID", job_ids_all)

        if st.button("Prepare download"):
            selected_job = next(
                (j for j in jobs if j["job_id"] == selected_job_id),
                None,
            )

            try:
                data = download_output_video(selected_job_id)
            except ClientError as ce:
                code = ce.response.get("Error", {}).get("Code")
                if code == "NoSuchKey":
                    st.error(
                        "Result video for this job is not ready yet "
                        "(result.mp4 not found in S3). Please wait and refresh."
                    )
                else:
                    st.error(f"Cannot download result: {ce}")
            else:
                # ตั้งชื่อไฟล์ให้ดูรู้เรื่อง
                if selected_job:
                    base = selected_job.get("original_filename") or selected_job_id
                    base = os.path.splitext(os.path.basename(base))[0]
                    mode_str = selected_job.get("mode", "dots")
                    download_name = f"{base}_{mode_str}.mp4"
                else:
                    download_name = f"{selected_job_id}_result.mp4"

                st.download_button(
                    label=f"Download {download_name}",
                    data=data,
                    file_name=download_name,
                    mime="video/mp4",
                )
