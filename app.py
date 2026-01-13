import os
import io
import json
import uuid
import datetime as dt

import boto3
import streamlit as st

# ---------------------------------------------------------
# S3 CONFIG (ต้องตรงกับ worker.py / config.py)
# ---------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
AWS_BUCKET = os.environ.get("S3_BUCKET")  # ใน Render ตั้งชื่อ S3_BUCKET ไว้แล้ว

PENDING_PREFIX = "jobs/pending/"
OUTPUT_PREFIX = "jobs/output/"
FAILED_PREFIX = "jobs/failed/"

# ---------------------------------------------------------
# S3 CLIENT
# ---------------------------------------------------------
session = boto3.session.Session(
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION,
)
s3 = session.client("s3")


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def generate_job_id() -> str:
    """สร้าง job id แบบ 20260113_125711__92fddc"""
    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}__{rand}"


def upload_video_and_create_job(video_file, mode: str):
    """
    1) อัปโหลดวิดีโอไปที่ jobs/pending/<job_id>/input/input.mp4
    2) สร้าง job JSON: jobs/pending/<job_id>.json
    3) คืนค่า job_id, video_key, job_key
    """
    job_id = generate_job_id()

    # ที่อยู่ของไฟล์ใน S3 (ต้องใช้ key นี้ใน worker ด้วย)
    video_key = f"{PENDING_PREFIX}{job_id}/input/input.mp4"
    job_key = f"{PENDING_PREFIX}{job_id}.json"

    # 1) อัปโหลดวิดีโอ
    file_bytes = video_file.read()
    video_buffer = io.BytesIO(file_bytes)

    s3.upload_fileobj(
        video_buffer,
        AWS_BUCKET,
        video_key,
        ExtraArgs={"ContentType": "video/mp4"},
    )

    # 2) สร้าง job JSON ให้ worker อ่าน
    job_data = {
        "job_id": job_id,
        "mode": mode,
        "video_key": video_key,  # สำคัญ! worker จะใช้ key นี้ download_file
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "status": "pending",
    }

    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=job_key,
        Body=json.dumps(job_data).encode("utf-8"),
        ContentType="application/json",
    )

    return job_id, video_key, job_key


def get_job_result(job_id: str):
    """ลองอ่านผลลัพธ์ ถ้า worker ทำเสร็จแล้วจะอยู่ที่ output หรือ failed"""
    output_key = f"{OUTPUT_PREFIX}{job_id}.json"
    failed_key = f"{FAILED_PREFIX}{job_id}.json"

    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=output_key)
        body = obj["Body"].read().decode("utf-8")
        return "done", json.loads(body)
    except s3.exceptions.NoSuchKey:
        pass
    except Exception:
        pass

    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=failed_key)
        body = obj["Body"].read().decode("utf-8")
        return "failed", json.loads(body)
    except s3.exceptions.NoSuchKey:
        return "pending", None
    except Exception:
        return "unknown", None


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI People Reader — Job maker", layout="wide")

st.title("AI People Reader — สร้างงานบน S3")

st.markdown(
    """
เว็บนี้มีหน้าที่อย่างเดียว:

1. อัปโหลดวิดีโอไปเก็บใน S3 ภายใต้โฟลเดอร์ `jobs/pending/`
2. สร้างไฟล์ job `.json` ให้ **worker** (src/worker.py) ไปอ่าน
"""
)

mode = st.selectbox("เลือกโหมดประมวลผล", ["dots", "skeleton", "effort"], index=0)

uploaded = st.file_uploader(
    "อัปโหลดวิดีโอ (mp4 / mov / m4v)", type=["mp4", "mov", "m4v"]
)

if st.button("สร้าง Job ใหม่ใน S3"):
    if uploaded is None:
        st.error("กรุณาอัปโหลดวิดีโอก่อนค่ะ")
    else:
        try:
            job_id, video_key, job_key = upload_video_and_create_job(uploaded, mode)

            st.success("สร้าง job ใหม่เรียบร้อยแล้วค่ะ 🎉")
            st.code(job_id, language="text")

            st.write("**Video S3 key:**")
            st.code(video_key, language="text")

            st.write("**Job JSON S3 key (อยู่ในโฟลเดอร์ jobs/pending/):**")
            st.code(job_key, language="text")

            st.info(
                "worker (src/worker.py) จะอ่าน job จากโฟลเดอร์ `jobs/pending/` "
                "แล้วเขียนผลลัพธ์ไปที่ `jobs/output/` หรือ `jobs/failed/`"
            )
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสร้าง job: {e}")


st.markdown("---")
st.subheader("เช็คสถานะผลลัพธ์ของ job")

check_id = st.text_input("กรอก Job ID (เช่น 20260113_125711__92fddc)")

if st.button("เช็คผลลัพธ์จาก S3"):
    if not check_id.strip():
        st.warning("ใส่ Job ID ก่อนนะคะ")
    else:
        status, data = get_job_result(check_id.strip())
        st.write(f"สถานะ: **{status}**")
        if data is not None:
            st.json(data)
