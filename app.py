# app.py  --- Streamlit frontend สำหรับสร้าง / เช็คงาน dot
import os
import io
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

st.set_page_config(page_title="AI People Reader - Job Manager", layout="wide")

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def new_job_id() -> str:
    """สร้าง job id ใหม่"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"

def upload_bytes_to_s3(data: bytes, bucket: str, key: str):
    s3.put_object(Bucket=bucket, Key=key, Body=data)

def upload_fileobj_to_s3(file_obj, bucket: str, key: str):
    s3.upload_fileobj(file_obj, bucket, key)

def get_json_from_s3(bucket: str, key: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except s3.exceptions.NoSuchKey:
        return None
    body = obj["Body"].read()
    return json.loads(body)

# ----------------------------------------------------------
# UI – อัปโหลดและสร้าง job ใหม่
# ----------------------------------------------------------
st.title("AI People Reader – Job Creator / Checker")

st.header("1. สร้าง Job ใหม่ใน S3")

mode = st.selectbox("เลือกโหมดการประมวลผล", ["dots"], index=0)

uploaded = st.file_uploader(
    "อัปโหลดวิดีโอ (MP4 / MOV / AVI)",
    type=["mp4", "mov", "avi", "m4v"]
)

if uploaded is not None:
    st.info(f"ไฟล์: **{uploaded.name}**, ขนาด ~{uploaded.size/1_000_000:.2f} MB")

if st.button("สร้าง Job ใหม่ใน S3", disabled=(uploaded is None)):
    if uploaded is None:
        st.warning("กรุณาอัปโหลดวิดีก่อนนะคะ")
    else:
        job_id = new_job_id()

        # 1) อัปโหลดวิดีโอไปที่ jobs/pending/<job_id>/input/input.mp4
        video_key = f"jobs/pending/{job_id}/input/input.mp4"
        upload_fileobj_to_s3(uploaded, AWS_BUCKET, video_key)

        # 2) สร้างไฟล์ job metadata ที่ jobs/pending/<job_id>.json
        job_meta = {
            "job_id": job_id,
            "mode": mode,
            "video_key": video_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }
        buf = io.BytesIO(json.dumps(job_meta, ensure_ascii=False).encode("utf-8"))
        meta_key = f"jobs/pending/{job_id}.json"
        upload_fileobj_to_s3(buf, AWS_BUCKET, meta_key)

        st.success("สร้าง job ใหม่เรียบร้อยแล้ว 🎉")
        st.code(job_id, language="text")
        st.write("Job JSON S3 key:", meta_key)
        st.write("Video S3 key:", video_key)

# ----------------------------------------------------------
# UI – เช็คสถานะของ job
# ----------------------------------------------------------
st.header("2. เช็คสถานะผลลัพธ์ของ job")

default_job_id = ""
input_job_id = st.text_input("กรอก Job ID (เช่น 20260113_133856__02d9f4)", value=default_job_id)

if st.button("เช็คผลลัพธ์จาก S3"):
    job_id = input_job_id.strip()
    if not job_id:
        st.warning("กรุณากรอก Job ID ก่อนนะคะ")
    else:
        # ลำดับการเช็ค:
        # 1) output -> done
        # 2) failed -> failed
        # 3) ถ้าไม่มีทั้งคู่ -> pending/processing
        output_key = f"jobs/output/{job_id}.json"
        failed_key = f"jobs/failed/{job_id}.json"

        job_output = get_json_from_s3(AWS_BUCKET, output_key)
        if job_output is not None:
            st.success("สถานะ: done ✅")
            st.json(job_output, expanded=False)
        else:
            job_failed = get_json_from_s3(AWS_BUCKET, failed_key)
            if job_failed is not None:
                st.error("สถานะ: failed ❌")
                st.json(job_failed, expanded=False)
            else:
                st.warning("ยังไม่พบผลลัพธ์ใน output หรือ failed\nอาจจะยังอยู่ในคิว (pending / processing)")
