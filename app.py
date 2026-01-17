# app.py — AI People Reader Job Manager (Johansson dots)

import os
import io
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "video/mp4") -> None:
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
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


def list_jobs() -> List[Dict[str, Any]]:
    """
    รวม jobs จาก pending / processing / finished / failed
    แล้วเรียงตาม created_at_utc จากใหม่ไปเก่า
    """
    jobs: List[Dict[str, Any]] = []

    def load_prefix(prefix: str) -> None:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=AWS_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".json"):
                    continue
                try:
                    job = s3_get_json(key)
                    jobs.append(job)
                except Exception:
                    # ถ้าอ่านไม่ได้ก็ข้ามไป
                    continue

    for prefix in (PENDING_PREFIX, PROCESSING_PREFIX, FINISHED_PREFIX, FAILED_PREFIX):
        load_prefix(prefix)

    def sort_key(j: Dict[str, Any]) -> str:
        return j.get("created_at_utc", "")

    jobs.sort(key=sort_key, reverse=True)
    return jobs


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    try:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": AWS_BUCKET, "Key": key},
            ExpiresIn=expires_in,
        )
    except ClientError:
        return ""


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI People Reader - Job Manager (App-maker-App-maker)",
    layout="wide",
)

st.title("🎬 AI People Reader - Job Manager (App-maker-App-maker)")

col_left, col_right = st.columns([1, 2])

# ---------------------------------------------------------------------------
# ① Create New Job
# ---------------------------------------------------------------------------

with col_left:
    st.markdown("## ① Create New Job")

    MODE_OPTIONS = {
        "Johansson dots – 1 person (แนะนำ)": "dots",
        "Johansson dots – 2 persons (ทดลอง)": "dots_2p",
    }

    mode_label = st.selectbox("Mode", list(MODE_OPTIONS.keys()))
    mode_value = MODE_OPTIONS[mode_label]

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v", "mpeg4"],
        help="Limit ~1GB per file",
    )

    user_note = st.text_input("Note (optional)")

    create_btn = st.button("Create job", type="primary")

    if create_btn:
        if uploaded_file is None:
            st.error("กรุณาอัปโหลดไฟล์วิดีโอก่อน")
        else:
            try:
                file_bytes = uploaded_file.read()
                job_id = new_job_id()

                # จัดเก็บ input video
                input_key = f"{PENDING_PREFIX}/{job_id}/input.mp4"
                upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

                # สามารถปรับเป็นชื่อไฟล์อื่นได้ถ้าต้องการ
                output_key = f"{OUTPUT_PREFIX}/{job_id}/result.mp4"

                now_iso = utc_now_iso()
                job = {
                    "job_id": job_id,
                    "status": "pending",
                    "mode": mode_value,
                    "input_key": input_key,
                    "output_key": output_key,
                    "created_at_utc": now_iso,
                    "updated_at_utc": now_iso,
                    "error": None,
                    "user_note": user_note or "",
                    "original_filename": uploaded_file.name,
                }

                job_json_key = f"{PENDING_PREFIX}/{job_id}.json"
                s3_put_json(job_json_key, job)

                st.success(
                    f"สร้างงานเรียบร้อยแล้ว 🎉\n\n"
                    f"**Job ID:** `{job_id}`\n\n"
                    f"กรุณาจด Job ID นี้ไว้เพื่อใช้เช็คสถานะและดาวน์โหลดผลลัพธ์"
                )
            except Exception as exc:
                st.error(f"เกิดข้อผิดพลาดขณะสร้างงาน: {exc}")


# ---------------------------------------------------------------------------
# ② Job List & Download
# ---------------------------------------------------------------------------

with col_right:
    st.markdown("## ② Job List & Download")

    if st.button("🔄 Refresh job list"):
        st.experimental_rerun()

    jobs = list_jobs()

    if not jobs:
        st.info("ยังไม่มีงานในระบบ")
    else:
        import pandas as pd

        rows = []
        for j in jobs:
            rows.append(
                {
                    "job_id": j.get("job_id"),
                    "status": j.get("status"),
                    "mode": j.get("mode"),
                    "created_at": j.get("created_at_utc"),
                    "updated_at": j.get("updated_at_utc"),
                    "error": j.get("error"),
                    "note": j.get("user_note"),
                    "file": j.get("original_filename"),
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("### ⬇️ Download processed video")

    job_id_for_dl = st.text_input("Enter Job ID to download result")

    if st.button("Download processed video"):
        if not job_id_for_dl.strip():
            st.error("กรุณาใส่ Job ID")
        else:
            job_json_key = f"{FINISHED_PREFIX}/{job_id_for_dl.strip()}.json"
            try:
                job = s3_get_json(job_json_key)
            except ClientError:
                st.error("ไม่พบงานสถานะ finished สำหรับ Job ID นี้")
            except Exception as exc:
                st.error(f"อ่านข้อมูลงานไม่สำเร็จ: {exc}")
            else:
                output_key = job.get("output_key")
                if not output_key:
                    st.error("งานนี้ไม่มี output_key ใน JSON")
                else:
                    url = generate_presigned_url(output_key)
                    if not url:
                        st.error("ไม่สามารถสร้างลิงก์ดาวน์โหลดได้")
                    else:
                        st.success("กดลิงก์ด้านล่างเพื่อดาวน์โหลดวิดีโอที่ประมวลผลแล้ว:")
                        st.markdown(f"[🎥 Download processed video]({url})")
