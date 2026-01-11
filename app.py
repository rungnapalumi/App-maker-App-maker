import os
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3
from botocore.config import Config

st.set_page_config(page_title="AI People Reader — Worker Mode", layout="wide")

REQUIRED_ENV = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "S3_BUCKET"]

def require_env():
    missing = [k for k in REQUIRED_ENV if not (os.getenv(k) or "").strip()]
    if missing:
        st.error("Missing env: " + ", ".join(missing))
        st.info("Render → Web service → Environment: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION / S3_BUCKET")
        st.stop()

AWS_REGION = (os.getenv("AWS_REGION") or "ap-southeast-1").strip()
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()

def s3():
    # No endpoint_url, no proxies
    cfg = Config(proxies={}, retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=cfg,
        aws_access_key_id=(os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
        aws_secret_access_key=(os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip(),
    )

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def put_json(key: str, data: dict):
    s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

def get_json(key: str) -> dict:
    obj = s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def exists(key: str) -> bool:
    try:
        s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except Exception:
        return False

def presigned_get(key: str, expires=3600) -> str:
    return s3().generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=expires,
    )

# ---------------- UI ----------------
st.sidebar.title("AI People Reader")
st.sidebar.selectbox("Mode", ["Worker mode (Background processing)"], index=0)

require_env()

st.title("🧠 AI People Reader — Worker Mode")
try:
    s3().head_bucket(Bucket=S3_BUCKET)
    st.success("S3 reachable ✅")
except Exception as e:
    st.error("S3 not reachable ❌")
    st.exception(e)
    st.stop()

st.caption("Upload video → Submit → Worker processes → Status updates → Download overlay (Johansson dots, dot=5, no audio)")

video = st.file_uploader("Upload video for background processing", type=["mp4", "mov", "m4v"])

colL, colR = st.columns([1, 1])

with colL:
    if video and st.button("Submit to worker"):
        with st.spinner("Uploading + creating job ticket… (large files may take a while)"):
            job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "__" + uuid.uuid4().hex[:6]

            input_key = f"jobs/{job_id}/input/input.mp4"
            status_key = f"jobs/{job_id}/status.json"
            pending_key = f"jobs/pending/{job_id}.json"

            # 1) upload video
            s3().upload_fileobj(video, S3_BUCKET, input_key)

            # 2) write status.json (queued)
            put_json(status_key, {
                "job_id": job_id,
                "status": "queued",
                "progress": 0,
                "message": "Queued",
                "updated_at": now_iso(),
                "outputs": {}
            })

            # 3) *** THIS IS THE MISSING PIECE ***
            #    write jobs/pending/<job_id>.json for worker to pick up
            put_json(pending_key, {
                "job_id": job_id,
                "input_key": input_key,
                "mode": "johansson_dots",
                "dot_radius": 5,
                "remove_audio": True,
                "created_at": now_iso(),
            })

        st.session_state["job_id"] = job_id
        st.success(f"Job submitted: {job_id}")
        st.info("Now worker should pick it up automatically. Refresh status on the right.")

with colR:
    st.subheader("Job status")
    job_id = st.session_state.get("job_id")
    if not job_id:
        st.info("Submit a job to see status here.")
    else:
        st.code(job_id)
        status_key = f"jobs/{job_id}/status.json"
        if exists(status_key):
            status = get_json(status_key)
            st.json(status)
            st.progress(int(status.get("progress", 0)))

            if status.get("status") == "done":
                overlay_key = (status.get("outputs") or {}).get("overlay_key")
                if overlay_key:
                    url = presigned_get(overlay_key)
                    st.markdown(f"[⬇️ Download overlay]({url})")
                else:
                    st.warning("Done but overlay_key missing.")
        else:
            st.warning("status.json not found yet. Please refresh.")
