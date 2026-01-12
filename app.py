import os
import json
import uuid
import time
from datetime import datetime, timezone

import streamlit as st
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI People Reader — Worker Mode", layout="wide")

# ----------------------------
# Required ENV
# ----------------------------
REQUIRED_ENV = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"]

def env(name: str) -> str:
    return (os.getenv(name) or "").strip()

AWS_REGION = env("AWS_REGION") or "ap-southeast-1"
S3_BUCKET = env("S3_BUCKET")

def require_env_or_stop():
    missing = [k for k in REQUIRED_ENV if not env(k)]
    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
        st.info(
            "Go to Render → Web Service → Environment and set:\n"
            "- AWS_ACCESS_KEY_ID\n"
            "- AWS_SECRET_ACCESS_KEY\n"
            "- AWS_REGION (ap-southeast-1)\n"
            "- S3_BUCKET (ai-people-reader-storage)"
        )
        st.stop()

# ----------------------------
# S3 client (no proxies)
# ----------------------------
def s3():
    cfg = Config(proxies={}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env("AWS_SECRET_ACCESS_KEY"),
    )

def s3_put_json(key: str, data: dict):
    s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def s3_get_json(key: str) -> dict:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def s3_exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False

def presigned_download_url(key: str, expires_seconds: int = 3600) -> str:
    return s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires_seconds,
    )

def now_iso():
    return datetime.now(timezone.utc).isoformat()

# ----------------------------
# UI
# ----------------------------
require_env_or_stop()

st.sidebar.title("AI People Reader")
mode = st.sidebar.selectbox("Mode", ["Worker mode (Background processing)"], index=0)

st.title("🧠 AI People Reader — Worker Mode")
st.caption("Upload video → Submit → Worker processes → Status updates → Download dots overlay (dot=5, no audio)")

# S3 health check
try:
    s3().head_bucket(Bucket=S3_BUCKET)
    st.success("S3 reachable ✅")
except Exception as e:
    st.error("S3 not reachable ❌")
    st.exception(e)
    st.stop()

# Layout
left, right = st.columns([1.05, 0.95], gap="large")

# Keep job_id in session
if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""

# ----------------------------
# LEFT: Upload + Submit
# ----------------------------
with left:
    st.subheader("Upload video for background processing")

    video = st.file_uploader(
        "Drag & drop or browse (MP4/MOV/M4V)",
        type=["mp4", "mov", "m4v"],
        accept_multiple_files=False
    )

    dot_radius = st.number_input("Dot radius (px)", min_value=1, max_value=15, value=5, step=1)
    dot_count = st.number_input("Dot count", min_value=30, max_value=500, value=150, step=10)

    submit = st.button("Submit to worker", disabled=(video is None))

    if submit and video is not None:
        with st.spinner("Uploading video + creating job ticket… (large files may take some time)"):
            job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "__" + uuid.uuid4().hex[:6]
            st.session_state["job_id"] = job_id

            # S3 keys
            input_key = f"jobs/{job_id}/input/input.mp4"
            status_key = f"jobs/{job_id}/status.json"
            pending_key = f"jobs/pending/{job_id}.json"

            # 1) Upload input video
            s3().upload_fileobj(video, S3_BUCKET, input_key)

            # 2) Write initial status.json (queued)
            s3_put_json(status_key, {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "updated_at": now_iso(),
                "outputs": {}
            })

            # 3) Write pending job ticket (THIS triggers worker)
            s3_put_json(pending_key, {
                "job_id": job_id,
                "input_key": input_key,
                "dot_radius": int(dot_radius),
                "dot_count": int(dot_count),
                "created_at": now_iso(),
            })

        st.success(f"Job submitted: {job_id}")
        st.info("Worker should pick it up automatically. Check status on the right.")

# ----------------------------
# RIGHT: Status + Auto Download Link
# ----------------------------
with right:
    st.subheader("Job status")

    job_id = st.session_state.get("job_id", "")
    job_id_input = st.text_input("Job ID", value=job_id, help="Paste a job_id to check status again later.")
    if job_id_input != job_id:
        st.session_state["job_id"] = job_id_input.strip()
        job_id = st.session_state["job_id"]

    if not job_id:
        st.info("Submit a job to see status here.")
    else:
        status_key = f"jobs/{job_id}/status.json"
        output_key = f"jobs/{job_id}/output/dots.mp4"

        # Refresh button
        colA, colB = st.columns([1, 1])
        with colA:
            refresh = st.button("Refresh status")
        with colB:
            auto_refresh = st.toggle("Auto refresh (every 5s)", value=False)

        def render_status_panel():
            # status.json
            if s3_exists(status_key):
                status = s3_get_json(status_key)
                st.code(job_id)
                st.json(status)
                st.progress(int(status.get("progress", 0)))
            else:
                st.warning("status.json not found yet. (Try refresh)")

            # Auto download link when output exists
            if s3_exists(output_key):
                url = presigned_download_url(output_key, expires_seconds=3600)
                st.success("✅ Processing completed — dots.mp4 is ready!")
                st.markdown(f"⬇️ **[Download dots video]({url})**")
            else:
                st.info("Output not ready yet. (Waiting for jobs/<job_id>/output/dots.mp4)")

        if refresh:
            render_status_panel()
        else:
            render_status_panel()

        # Optional auto refresh loop
        if auto_refresh:
            st.caption("Auto-refresh is ON. Turn it off anytime.")
            # simple loop with limited iterations to avoid infinite blocking
            for _ in range(60):  # up to ~5 minutes
                time.sleep(5)
                st.experimental_rerun()
