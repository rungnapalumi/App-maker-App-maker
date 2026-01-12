import streamlit as st
import boto3
import json
import time
import os
from datetime import datetime

# =========================
# CONFIG
# =========================
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
S3_BUCKET = os.environ.get("S3_BUCKET")

POLL_INTERVAL = 5  # seconds

# =========================
# AWS CLIENT
# =========================
s3 = boto3.client("s3", region_name=AWS_REGION)

# =========================
# HELPERS
# =========================
def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False


def presigned_url(key: str, expires=3600):
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires,
    )


def read_status(job_id: str):
    key = f"jobs/{job_id}/status.json"
    if not s3_exists(key):
        return None
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


# =========================
# UI
# =========================
st.set_page_config(page_title="AI People Reader – Worker Mode", layout="wide")

st.title("🧠 AI People Reader — Worker Mode")
st.caption(
    "Upload video → Submit → Worker processes → Status updates → Download overlay "
    "(Johansson dots, dot=5, no audio)"
)

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader(
    "Upload video for background processing",
    type=["mp4", "mov", "m4v", "mpeg4"],
)

dot_radius = st.number_input("Dot radius (px)", 1, 20, 5)
dot_count = st.number_input("Dot count", 5, 200, 30)

submit = st.button("Submit to worker", disabled=uploaded is None)

# =========================
# SUBMIT JOB
# =========================
if submit and uploaded:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    job_id = f"{ts}__{os.urandom(3).hex()}"

    job_prefix = f"jobs/{job_id}"
    input_key = f"{job_prefix}/input/input.mp4"
    status_key = f"{job_prefix}/status.json"

    # Upload video
    s3.upload_fileobj(uploaded, S3_BUCKET, input_key)

    # Create status.json
    status = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "dot_radius": dot_radius,
        "dot_count": dot_count,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "outputs": {},
    }

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=status_key,
        Body=json.dumps(status).encode("utf-8"),
        ContentType="application/json",
    )

    st.session_state["job_id"] = job_id
    st.success(f"Job submitted: {job_id}")

# =========================
# JOB STATUS
# =========================
job_id = st.session_state.get("job_id")

st.subheader("Job status")

if job_id:
    st.code(job_id)

    auto = st.toggle("Auto refresh (every 5s)", value=True)

    status_box = st.empty()
    download_box = st.empty()

    while True:
        status = read_status(job_id)

        if status:
            status_box.json(status)

            output_key = f"jobs/{job_id}/output/dots.mp4"

            if s3_exists(output_key):
                url = presigned_url(output_key)
                download_box.success("✅ Output ready")
                download_box.markdown(
                    f"""
                    ### ⬇ Download result
                    [Download dots.mp4]({url})
                    """,
                    unsafe_allow_html=True,
                )
                break

        else:
            status_box.info("Waiting for status.json ...")

        if not auto:
            break

        time.sleep(POLL_INTERVAL)
        st.experimental_rerun()
else:
    st.info("Submit a job to see status here.")
