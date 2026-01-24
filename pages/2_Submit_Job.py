import os
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3
from botocore.exceptions import ClientError


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Submit Job (S3)", layout="wide")
st.title("🚀 Submit Job to S3 (Safe / Separate Page)")

st.caption(
    "หน้านี้เป็นหน้าใหม่แยกจาก app.py เดิม: ทำแค่อัปโหลด + สร้าง job.json/status.json ใน S3 (ไม่ยุ่งโค้ดเดิม)"
)

# =========================
# Env
# =========================
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

with st.expander("🔧 Environment (read-only)", expanded=False):
    st.write("AWS_BUCKET =", AWS_BUCKET)
    st.write("AWS_REGION =", AWS_REGION)

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)


# =========================
# Helpers
# =========================
def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}"


def s3_put_json(key: str, obj: dict):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def s3_put_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def guess_content_type(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".mov"):
        return "video/quicktime"
    return "application/octet-stream"


def build_job_manifest(job_id: str, input_key: str, modes: list[str], note: str = "") -> dict:
    return {
        "job_id": job_id,
        "input_key": input_key,
        "modes": modes,
        "note": note,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "submit-v1",
    }


# =========================
# UI: Submit
# =========================
st.subheader("1) Upload video + create job.json")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "webm"])
    note = st.text_input("Note (optional)", value="")

with col2:
    st.markdown("### Modes to request")
    mode_overlay = st.checkbox("overlay", value=True)
    mode_dots = st.checkbox("dots", value=False)
    mode_skeleton = st.checkbox("skeleton", value=False)
    mode_report = st.checkbox("report", value=False)

modes = []
if mode_overlay:
    modes.append("overlay")
if mode_dots:
    modes.append("dots")
if mode_skeleton:
    modes.append("skeleton")
if mode_report:
    modes.append("report")

st.caption("modes จะถูกเขียนลง jobs/<job_id>/job.json เพื่อให้ worker อ่านไปทำงาน")


if st.button("🚀 Submit job", disabled=(uploaded is None)):
    try:
        job_id = new_job_id()

        filename = uploaded.name if uploaded else "input.mp4"
        content_type = guess_content_type(filename)

        # 1) upload input video to S3
        input_key = f"jobs/{job_id}/input/{filename}"
        video_bytes = uploaded.getvalue()
        s3_put_bytes(input_key, video_bytes, content_type=content_type)

        # 2) write job manifest
        job = build_job_manifest(job_id, input_key, modes=modes, note=note)
        s3_put_json(f"jobs/{job_id}/job.json", job)

        # 3) initial status
        s3_put_json(f"jobs/{job_id}/status.json", {"status": "queued", "job_id": job_id})

        st.success("Submitted ✅")
        st.code(json.dumps(job, ensure_ascii=False, indent=2))

        st.markdown("### Next")
        st.write("✅ ตอนนี้มี job ใน S3 แล้ว: `jobs/{job_id}/...`")

        # Link ideas
        st.markdown("**Open results (choose one):**")
        pres_url = st.text_input(
            "Presentation Analysis base URL (optional)",
            value="",
            placeholder="e.g. https://presentation-analysis.onrender.com",
        )
        if pres_url.strip():
            st.link_button("Open in Presentation Analysis", f"{pres_url.rstrip('/')}/?job_id={job_id}")

        st.info(
            "ถ้า worker ทำงานอยู่และอ่าน jobs/<job_id>/job.json ได้ "
            "มันจะเขียน output ลง jobs/<job_id>/output/... แล้วคุณไปเปิดดูได้ในหน้า S3 Browser"
        )

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)


st.divider()
st.subheader("2) Verify job exists (read-only)")

job_id_check = st.text_input("Job ID to check", value="")
if st.button("Check status.json"):
    if not job_id_check.strip():
        st.warning("Please enter job_id")
    else:
        key = f"jobs/{job_id_check.strip()}/status.json"
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            data = obj["Body"].read().decode("utf-8", errors="replace")
            st.json(json.loads(data))
        except ClientError as e:
            st.error("Cannot read status.json")
            st.exception(e)
        except Exception as e:
            st.error("Failed to parse status.json")
            st.exception(e)
