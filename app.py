# app.py — App-maker-App-maker (Submit job + Verify status + Download buttons)
# ✅ Added: auto-fill latest job_id in Verify section (st.session_state["last_job_id"])

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import boto3
import streamlit as st


# -----------------------------
# Config
# -----------------------------
AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

ROOT_PREFIX = "jobs"  # jobs/<job_id>/...


# -----------------------------
# Helpers
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}"


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    raw = obj["Body"].read()
    return json.loads(raw.decode("utf-8"))


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def upload_fileobj_to_s3(fileobj, key: str, content_type: str = "video/mp4") -> None:
    s3.upload_fileobj(
        fileobj,
        AWS_BUCKET,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def presigned_get_url(key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=expires,
    )


def guess_content_type_from_key(key: str) -> str:
    k = key.lower()
    if k.endswith(".mp4"):
        return "video/mp4"
    if k.endswith(".pdf"):
        return "application/pdf"
    if k.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI People Reader - Submit Job", layout="wide")
st.sidebar.title("app")
st.sidebar.page_link("app.py", label="Submit Job", icon="🧪")

st.title("1) Upload video + create job.json")

with st.expander("🔧 Environment", expanded=False):
    st.write(
        {
            "AWS_BUCKET": AWS_BUCKET,
            "AWS_REGION": AWS_REGION,
            "ROOT_PREFIX": ROOT_PREFIX,
        }
    )

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    uploaded = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "m4v", "webm", "mpeg4"],
        help="Limit depends on your Streamlit/Render setup",
    )
    note = st.text_input("Note (optional)", value="")

with col_right:
    st.subheader("Modes to request")
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

submit_disabled = (uploaded is None) or (len(modes) == 0)

if st.button("🚀 Submit job", disabled=submit_disabled):
    try:
        job_id = new_job_id()
        safe_name = uploaded.name.replace("/", "_").replace("\\", "_")

        input_key = f"{ROOT_PREFIX}/{job_id}/input/{safe_name}"
        job_key = f"{ROOT_PREFIX}/{job_id}/job.json"
        status_key = f"{ROOT_PREFIX}/{job_id}/status.json"

        # Upload input video
        uploaded.seek(0)
        upload_fileobj_to_s3(
            uploaded,
            input_key,
            content_type=guess_content_type_from_key(input_key),
        )

        # Create job.json
        job_payload = {
            "job_id": job_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "note": note,
            "modes": modes,
            "input_key": input_key,
            "params": {},
        }
        s3_put_json(job_key, job_payload)

        # Create status.json (queued)
        status_payload = {
            "status": "queued",
            "job_id": job_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "outputs": {},
        }
        s3_put_json(status_key, status_payload)

        # ✅ Save latest job_id for auto-fill in Verify section
        st.session_state["last_job_id"] = job_id

        st.success("Submitted!")
        st.code(job_id)
        st.info("เลื่อนลงไปกด Check ได้เลย (Job ID จะถูกใส่ให้อัตโนมัติ)")

    except Exception as e:
        st.error(f"Submit failed: {e}")


st.divider()
st.title("2) Verify job exists (read-only)")

job_id_to_check = st.text_input(
    "Job ID to check",
    value=st.session_state.get("last_job_id", ""),
)

check = st.button("Check status.json")

if check:
    if not job_id_to_check.strip():
        st.warning("Please enter Job ID")
    else:
        jid = job_id_to_check.strip()
        status_key = f"{ROOT_PREFIX}/{jid}/status.json"

        if not s3_key_exists(status_key):
            st.error(f"Not found: {status_key}")
        else:
            try:
                status_data = s3_get_json(status_key)
                st.json(status_data)

                # ---- Download buttons ONLY ----
                outputs = (status_data or {}).get("outputs") or {}
                if outputs:
                    st.subheader("Downloads")
                    for name, s3_key in outputs.items():
                        if not s3_key:
                            continue
                        try:
                            url = presigned_get_url(s3_key, expires=3600)
                            if hasattr(st, "link_button"):
                                st.link_button(f"⬇️ Download {name}", url)
                            else:
                                st.markdown(f"[⬇️ Download {name}]({url})")
                        except Exception as e:
                            st.error(f"Cannot create download link for {name}: {e}")
                else:
                    st.caption("No outputs yet (worker ยังไม่เขียน outputs ลง status.json)")

            except Exception as e:
                st.error(f"Read status failed: {e}")
