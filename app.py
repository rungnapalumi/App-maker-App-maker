import os
import json
import uuid
import importlib
from datetime import datetime, timezone

import streamlit as st
import boto3
from botocore.exceptions import ClientError

st.set_page_config(page_title="AI People Reader", layout="wide")

# -----------------------------
# Helpers: env + S3 client
# (NO endpoint_url!)
# -----------------------------
REQUIRED_ENV = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"]

def require_env(keys):
    missing = [k for k in keys if not (os.getenv(k) or "").strip()]
    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
        st.info("Go to Render → Service → Environment and set these variables.")
        st.stop()

def s3():
    # ✅ IMPORTANT: do NOT pass endpoint_url
    return boto3.client(
        "s3",
        region_name=(os.getenv("AWS_REGION") or "ap-southeast-1").strip(),
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def put_json(bucket, key, data):
    s3().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )

def get_json(bucket, key):
    obj = s3().get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def exists(bucket, key):
    try:
        s3().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def presigned_get(bucket, key, expires=3600):
    return s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )

# -----------------------------
# UI: Mode
# -----------------------------
st.sidebar.title("AI People Reader")
mode = st.sidebar.selectbox(
    "Mode",
    ["Original app", "Worker mode (Background Processing)"],
    index=1,
)

# -----------------------------
# Mode 1: Original app
# -----------------------------
if mode == "Original app":
    st.info("Running: Original app (app_legacy.py)")
    try:
        legacy = importlib.import_module("app_legacy")
        # if legacy has main(), call it; else importing may run it
        if hasattr(legacy, "main") and callable(legacy.main):
            legacy.main()
    except ModuleNotFoundError:
        st.error("Cannot find app_legacy.py. Please rename old app.py → app_legacy.py first.")
    except Exception as e:
        st.error("Original app crashed:")
        st.exception(e)
    st.stop()

# -----------------------------
# Mode 2: Worker mode (S3 queue)
# -----------------------------
st.title("🧠 AI People Reader — Worker Mode")
require_env(REQUIRED_ENV)

bucket = (os.getenv("S3_BUCKET") or "").strip()

# Quick S3 check
try:
    s3().head_bucket(Bucket=bucket)
    st.success("S3 reachable ✅")
except Exception as e:
    st.error("S3 connection failed ❌")
    st.exception(e)
    st.stop()

st.write("Upload video → Submit → Worker processes → Status updates → Download overlay")

video = st.file_uploader("Upload video for background processing", type=["mp4", "mov", "m4v"])

col1, col2 = st.columns([1, 1])

with col1:
    if video and st.button("Submit to worker"):
        with st.spinner("Uploading to S3... (large files may take a while)"):
            job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + "_" + uuid.uuid4().hex[:6]
            input_key = f"jobs/{job_id}/input/input.mp4"

            # Upload input video
            s3().upload_fileobj(video, bucket, input_key)

            # Write initial status
            put_json(bucket, f"jobs/{job_id}/status.json", {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "updated_at": now_iso(),
                "outputs": {}
            })

            # Enqueue ticket
            put_json(bucket, f"jobs/pending/{job_id}.json", {
                "job_id": job_id,
                "input_key": input_key,
                "created_at": now_iso(),
            })

        st.session_state["job_id"] = job_id
        st.success(f"Job submitted: {job_id}")

with col2:
    job_id = st.session_state.get("job_id")
    st.subheader("Job status")
    if not job_id:
        st.info("Submit a job to see status here.")
    else:
        st.code(job_id)
        status_key = f"jobs/{job_id}/status.json"

        if exists(bucket, status_key):
            status = get_json(bucket, status_key)
            st.json(status)
            st.progress(int(status.get("progress", 0)))

            if status.get("status") == "done":
                overlay_key = (status.get("outputs") or {}).get("overlay_key") or (status.get("outputs") or {}).get("overlay_key")
                # some workers used overlay_key, others used overlay_key name
                if overlay_key:
                    url = presigned_get(bucket, overlay_key, expires=3600)
                    st.markdown(f"[⬇️ Download overlay]({url})")
                else:
                    st.warning("Done but no overlay_key found in outputs.")
        else:
            st.info("Waiting for status.json... (refresh)")
