import os
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3

st.set_page_config(page_title="AI People Reader", layout="wide")

# -----------------------------
# Env + S3 client (NO endpoint_url)
# -----------------------------
REQUIRED_ENV = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"]

def require_env():
    missing = [k for k in REQUIRED_ENV if not (os.getenv(k) or "").strip()]
    if missing:
        st.error("Missing environment variables: " + ", ".join(missing))
        st.info("Render → (Web service) → Environment: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION / S3_BUCKET")
        st.stop()

def s3():
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
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
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
# UI
# -----------------------------
st.sidebar.title("AI People Reader")
mode = st.sidebar.selectbox("Mode", ["Worker mode (Background Processing)"], index=0)

require_env()
bucket = (os.getenv("S3_BUCKET") or "").strip()

st.title("🧠 AI People Reader — Worker Mode")
try:
    s3().head_bucket(Bucket=bucket)
    st.success("S3 reachable ✅")
except Exception as e:
    st.error("S3 not reachable ❌")
    st.exception(e)
    st.stop()

st.markdown("Upload video → Submit → Worker processes → Status updates → Download overlay")

video = st.file_uploader("Upload video for background processing", type=["mp4", "mov", "m4v"])

col1, col2 = st.columns([1, 1])

with col1:
    if video and st.button("Submit to worker"):
        with st.spinner("Uploading to S3... (large files may take a while)"):
            job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "__" + uuid.uuid4().hex[:6]
            input_key = f"jobs/{job_id}/input/input.mp4"

            # Upload video to S3
            s3().upload_fileobj(video, bucket, input_key)

            # Initial status
            put_json(bucket, f"jobs/{job_id}/status.json", {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "updated_at": now_iso(),
                "outputs": {}
            })

            # Queue ticket (worker polls this)
            put_json(bucket, f"jobs/pending/{job_id}.json", {
                "job_id": job_id,
                "input_key": input_key,
                "created_at": now_iso()
            })

        st.session_state["job_id"] = job_id
        st.success(f"Job submitted: {job_id}")

with col2:
    st.subheader("Job status")
    job_id = st.session_state.get("job_id")
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
                overlay_key = (status.get("outputs") or {}).get("overlay_key")
                if overlay_key:
                    url = presigned_get(bucket, overlay_key, expires=3600)
                    st.markdown(f"[⬇️ Download overlay]({url})")
                else:
                    st.warning("Done but overlay_key missing.")
        else:
            st.info("Waiting for status.json... (refresh)")
