# app.py  (อยู่ที่ root ของ repo App-maker-App-maker)
#
# หน้าที่:
# 1. รับวิดีโอจากผู้ใช้ผ่าน Streamlit
# 2. อัปโหลดไฟล์ไปที่ S3 bucket (ใช้ AWS_BUCKET หรือ S3_BUCKET)
# 3. สร้าง job JSON ไว้ที่ jobs/pending/<job_id>.json
#    ให้ worker (src/worker.py) ไปดึงมาทำงานต่อ
#
# รูปแบบ S3 key ที่ใช้:
# - วิดีโออินพุต: jobs/pending/<job_id>/input/input.mp4
# - job JSON:     jobs/pending/<job_id>.json

import os
import json
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import streamlit as st


# -----------------------------
# ตั้งค่า S3 จาก Environment
# -----------------------------
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

# ใช้ AWS_BUCKET เป็นหลัก ถ้าไม่มีค่อย fallback ไปที่ S3_BUCKET
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")

if not AWS_BUCKET:
    st.error(
        "ไม่พบ environment variable ชื่อ AWS_BUCKET หรือ S3_BUCKET\n"
        "กรุณาตั้งค่าในหน้า Environment ของ Render ก่อนค่ะ"
    )
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)


# -----------------------------
# ฟังก์ชันสร้าง Job ใน S3
# -----------------------------
def create_job_in_s3(uploaded_file, mode: str = "dots"):
    """
    - อัปโหลดไฟล์ไป S3: jobs/pending/<job_id>/input/input.mp4
    - สร้าง job JSON:     jobs/pending/<job_id>.json
    - คืนค่า (job_id, input_key, job_key)
    """
    now = datetime.now(timezone.utc)

    # ตัวอย่าง job_id: 20260113_121211__db35b5
    job_id = now.strftime("%Y%m%d_%H%M%S") + "__" + uuid.uuid4().hex[:6]

    base_prefix = f"jobs/pending/{job_id}"
    input_key = f"{base_prefix}/input/input.mp4"
    job_key = f"jobs/pending/{job_id}.json"

    # อัปโหลดไฟล์วิดีโอไป S3
    uploaded_file.seek(0)
    try:
        s3.upload_fileobj(uploaded_file, AWS_BUCKET, input_key)
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"อัปโหลดวิดีโอไป S3 ไม่สำเร็จ: {e}") from e

    # เตรียมข้อมูล job ให้ worker ใช้
    job_data = {
        "job_id": job_id,
        "video_key": input_key,          # worker จะใช้ key นี้ไปดาวน์โหลดไฟล์
        "mode": mode,                    # ตอนนี้มี 'dots' อย่างเดียว
        "created_at": now.isoformat(),
    }

    try:
        s3.put_object(
            Bucket=AWS_BUCKET,
            Key=job_key,
            Body=json.dumps(job_data).encode("utf-8"),
            ContentType="application/json",
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"เขียนไฟล์ job JSON ไป S3 ไม่สำเร็จ: {e}") from e

    return job_id, input_key, job_key


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI People Reader – Job Creator",
    page_icon="✨",
    layout="centered",
)

st.title("AI People Reader – Video Job Creator")
st.write(
    "อัปโหลดวิดีโอ แล้วระบบจะสร้าง **job** ใหม่ไว้ใน S3 ที่ `jobs/pending/` "
    "จากนั้น worker จะมาหยิบไปประมวลผลเองค่ะ 🤖"
)

st.markdown("---")

uploaded_file = st.file_uploader(
    "1) เลือกวิดีโอที่ต้องการประมวลผล",
    type=["mp4", "mov", "avi", "mkv"],
    help="ไฟล์จะถูกอัปโหลดไปเก็บใน S3 bucket ของคุณ",
)

mode = st.selectbox(
    "2) เลือกโหมดการประมวลผล",
    ["dots"],
    help="ตอนนี้มีเฉพาะ Johansson dots (โหมด 'dots')",
)

create_btn = st.button("3) สร้าง Job ใหม่ใน S3")

if create_btn:
    if not uploaded_file:
        st.warning("กรุณาเลือกวิดีโอก่อนค่ะ")
    else:
        with st.spinner("กำลังอัปโหลดวิดีโอและสร้าง job บน S3 ..."):
            try:
                job_id, input_key, job_key = create_job_in_s3(uploaded_file, mode)
            except RuntimeError as e:
                st.error(str(e))
            else:
                st.success("สร้าง job ใหม่เรียบร้อยแล้วค่ะ 🎉")
                st.write("**Job ID:**")
                st.code(job_id, language="bash")

                st.write("**Video S3 key:**")
                st.code(input_key, language="bash")

                st.write("**Job JSON S3 key (อยู่ในโฟลเดอร์ jobs/pending/):**")
                st.code(job_key, language="bash")

                st.info(
                    "ฝั่ง worker (src/worker.py) จะอ่าน job จาก `jobs/pending/` แล้วเขียนผลลัพธ์ไปที่ "
                    f"`jobs/output/{job_id}/...` ให้เองค่ะ"
                )

st.markdown("---")
st.caption(
    f"ใช้ bucket: `{AWS_BUCKET}` | region: `{AWS_REGION}` | "
    "เว็บนี้เป็นฝั่งสร้าง job เท่านั้น งานประมวลผลจริงอยู่ใน background worker."
)
