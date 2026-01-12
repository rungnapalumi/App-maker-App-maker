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
# Helpers
# ----------------------------
def env(name: str) -> str:
    return (os.getenv(name) or "").strip()

AWS_REGION = env("AWS_REGION") or "ap-southeast-1"
S3_BUCKET = env("S3_BUCKET")

REQUIRED_ENV = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"]


def require_env_or_stop():
    missing = [k for k in REQUIRED_ENV if not env(k)]
    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
        st.info(
            "Render → Web Service → Environment:\n"
            "- AWS_ACCESS_KEY_ID\n"
            "- AWS_SECRET_ACCESS_KEY\n"
            "- AWS_REGION (ap-southeast-1)\n"
            "- S3_BUCKET (ai-people-reader-storage)"
        )
        st.stop()


def s3():
    cfg = Config(proxies={}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=env("AWS_SECRET_ACCESS_KEY"),
    )


def now_iso():
    return datetime.now(timezone.utc).isoformat()


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


def safe_rerun():
    """
    Streamlit API differs by version:
    - Newer: st.rerun()
    - Older: st.experimental_rerun()
    """
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # fallback: do nothing (manual refresh)
        pass


# ----------------------------
# UI
# ----------------------------
require_env_or_stop()

st.sidebar.title("AI People Reader")
st.sidebar.selectbox("Mode", ["Worker mode (Background processing)"], index=0)

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

# session state
if "job_id" not in st.session_state:
    st.session_state["job_id"] = ""

left, right = st.columns([1.05, 0.95], gap="large")

# ----------------------------
# LEFT: Upload + Submit
# ----------------------------
with left:
    st.subheader("Upload video for background processing")

    video = st.file_uploader(
        "Drag & drop or browse (MP4/MOV/M4V)",
        type=["mp4", "mov", "m4v", "mpeg4"],
        accept_multiple_files=False,
    )

    dot_radius = st.number_input("Dot radius (px)", min_value=1, max_value=15, value=5, step=1)
    dot_count = st.number_input("Dot count", min_value=5, max_value=500, value=30, step=5)

    submit = st.button("Submit to worker", disabled=(video is None))

    if submit and video is not None:
        with st.spinner("Uploading video + creating job ticket…"):
            job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "__" + uuid.uuid4().hex[:6]
            st.session_state["job_id"] = job_id

            input_key = f"jobs/{job_id}/input/input.mp4"
            status_key = f"jobs/{job_id}/status.json"
            pending_key = f"jobs/pending/{job_id}.json"

            # 1) Upload video
            s3().upload_fileobj(video, S3_BUCKET, input_key)

            # 2) Create status.json (queued)
            s3_put_json(status_key, {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "dot_radius": int(dot_radius),
                "dot_count": int(dot_count),
                "updated_at": now_iso(),
                "outputs": {}
            })

            # 3) Create pending job ticket (worker will pick it up)
            s3_put_json(pending_key, {
                "job_id": job_id,
                "input_key": input_key,
                "dot_radius": int(dot_radius),
                "dot_count": int(dot_count),
                "created_at": now_iso(),
            })

        st.success(f"Job submitted: {job_id}")
        st.info("Now worker should pick it up automatically. Refresh status on the right.")

# ----------------------------
# RIGHT: Status + Auto download link
# ----------------------------
with right:
    st.subheader("Job status")

    job_id = st.session_state.get("job_id", "").strip()

    job_id_input = st.text_input("Job ID", value=job_id, help="Paste a job_id to check status later.")
    if job_id_input.strip() != job_id:
        st.session_state["job_id"] = job_id_input.strip()
        job_id = st.session_state["job_id"]

    if not job_id:
        st.info("Submit a job to see status here.")
    else:
        status_key = f"jobs/{job_id}/status.json"
        output_key = f"jobs/{job_id}/output/dots.mp4"

        colA, colB = st.columns([1, 1])
        with colA:
            refresh = st.button("Refresh status")
        with colB:
            auto_refresh = st.toggle("Auto refresh (every 5s)", value=True)

        # Render panel once
        def render_panel():
            if s3_exists(status_key):
                status = s3_get_json(status_key)
                st.json(status)
                st.progress(int(status.get("progress", 0)))
            else:
                st.warning("status.json not found yet. Please wait a moment and refresh.")

            if s3_exists(output_key):
                url = presigned_download_url(output_key, expires_seconds=3600)
                st.success("✅ Output ready!")
                st.markdown(f"⬇️ **[Download dots.mp4]({url})**")
            else:
                st.info("Output not ready yet. (Waiting for jobs/<job_id>/output/dots.mp4)")

        # Manual refresh
        if refresh:
            render_panel()
        else:
            render_panel()

        # Auto refresh loop (no experimental_rerun)
        if auto_refresh:
            time.sleep(5)
            safe_rerun()
