# app.py — AI People Reader Job Manager (dots + skeleton) | NO AUDIO OPTIONS

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET (or S3_BUCKET) environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

st.set_page_config(page_title="AI People Reader - Job Manager", layout="wide")


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def upload_bytes_to_s3(data: bytes, key: str, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


def s3_get_json(key: str) -> Dict[str, Any]:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def presigned_get_url(key: str, expires: int = 3600) -> str:
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": AWS_BUCKET, "Key": key},
        ExpiresIn=expires,
    )


def create_job(file_bytes: bytes, mode: str, job_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upload input to:
      jobs/pending/<job_id>/input/input.mp4
    Save job json to:
      jobs/pending/<job_id>.json
    Output expected at:
      jobs/output/<job_id>/result.mp4
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    upload_bytes_to_s3(file_bytes, input_key, content_type="video/mp4")

    now = utc_now_iso()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,               # "dots" or "skeleton"
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
    }

    # top-level fields for worker
    for k, v in (job_fields or {}).items():
        job[k] = v

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)
    return job


def list_jobs() -> List[Dict[str, Any]]:
    all_jobs: List[Dict[str, Any]] = []
    pairs = [
        (JOBS_PENDING_PREFIX, "pending"),
        (JOBS_PROCESSING_PREFIX, "processing"),
        (JOBS_FINISHED_PREFIX, "finished"),
        (JOBS_FAILED_PREFIX, "failed"),
    ]

    for prefix, status in pairs:
        try:
            resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=prefix)
        except ClientError as e:
            st.error(f"List error {prefix}: {e}")
            continue

        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                j = s3_get_json(key)
            except ClientError:
                continue

            j["status"] = status
            j["s3_key"] = key
            all_jobs.append(j)

    all_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=False)
    return all_jobs


def download_output_video(job_id: str) -> bytes:
    key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return obj["Body"].read()


def _get_outputs_from_status_json(job_id: str) -> Optional[Dict[str, str]]:
    """
    รองรับระบบแบบใหม่ที่มี status.json:
      jobs/<job_id>/status.json
    โดยมีรูปแบบ:
      { ..., "outputs": { "overlay": "jobs/<job_id>/output/overlay.mp4", ... } }
    """
    status_key = f"jobs/{job_id}/status.json"
    if not s3_key_exists(status_key):
        return None
    try:
        status_data = s3_get_json(status_key)
        outputs = (status_data or {}).get("outputs") or {}
        if isinstance(outputs, dict):
            # ensure str->str
            cleaned: Dict[str, str] = {}
            for k, v in outputs.items():
                if isinstance(k, str) and isinstance(v, str) and v.strip():
                    cleaned[k] = v.strip()
            return cleaned if cleaned else None
    except Exception:
        return None
    return None


def _pick_best_output_key(job_id: str) -> str:
    """
    เลือก key สำหรับดาวน์โหลดโดย "ไม่เปลี่ยน UI":
    1) ถ้ามี jobs/<job_id>/status.json และมี outputs -> เลือกตัวที่ชื่อ result > overlay > ตัวแรก
    2) fallback เดิม: jobs/output/<job_id>/result.mp4
    """
    outputs = _get_outputs_from_status_json(job_id)
    if outputs:
        # priority
        for preferred in ("result", "overlay", "dots", "skeleton", "report"):
            if preferred in outputs:
                return outputs[preferred]
        # otherwise first item
        return next(iter(outputs.values()))
    return f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"


# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("AI People Reader - Job Manager (No Audio)")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("Create New Job")

    mode = st.selectbox("Mode", ["dots", "skeleton"], index=0)

    dot_radius = 5
    skeleton_color = "#00FF00"
    skeleton_thickness = 2

    if mode == "dots":
        st.subheader("Dots settings")
        dot_radius = st.slider("Dot size (radius px)", 1, 20, 5, 1)

    if mode == "skeleton":
        st.subheader("Skeleton settings")
        skeleton_color = st.color_picker("Line color (default green)", "#00FF00")
        skeleton_thickness = st.slider("Line thickness (px)", 1, 20, 2, 1)

    uploaded_file = st.file_uploader(
        "Upload video file",
        type=["mp4", "mov", "m4v"],
        accept_multiple_files=False,
    )

    if st.button("Create job"):
        if not uploaded_file:
            st.warning("Please upload a video file first.")
        else:
            job_fields: Dict[str, Any] = {}

            if mode == "dots":
                job_fields["dot_radius"] = int(dot_radius)

            if mode == "skeleton":
                job_fields["skeleton_line_color"] = str(skeleton_color)
                job_fields["skeleton_line_thickness"] = int(skeleton_thickness)

            job = create_job(uploaded_file.read(), mode, job_fields=job_fields)
            # ✅ จำ job ล่าสุด (ไม่เปลี่ยน UI, แค่ช่วยเลือกให้ตรงอัตโนมัติ)
            st.session_state["last_job_id"] = job["job_id"]

            st.success(f"Created job: {job['job_id']}")
            st.json(job)

with col_right:
    st.header("Jobs")

    if st.button("Refresh job list"):
        st.rerun()

    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet.")
    else:
        df = pd.DataFrame(
            [
                {
                    "job_id": j.get("job_id"),
                    "status": j.get("status"),
                    "mode": j.get("mode"),
                    "dot_radius": j.get("dot_radius"),
                    "skeleton_color": j.get("skeleton_line_color"),
                    "skeleton_thickness": j.get("skeleton_line_thickness"),
                    "created_at": j.get("created_at"),
                    "updated_at": j.get("updated_at"),
                    "error": j.get("error"),
                }
                for j in jobs
            ]
        )
        st.dataframe(df, use_container_width=True)

        st.subheader("Download result video ↪")

        job_ids_all = [j["job_id"] for j in jobs]

        # ✅ auto-select ล่าสุด ถ้ามี
        last_id = st.session_state.get("last_job_id")
        default_index = 0
        if last_id and last_id in job_ids_all:
            default_index = job_ids_all.index(last_id)

        selected_job_id = st.selectbox("Select job_id", job_ids_all, index=default_index)

        if st.button("Prepare download"):
            # ✅ "ไม่แตะ UI เดิม" แต่เพิ่มความสามารถ:
            #    - ถ้ามี status.json outputs -> ใช้อันนั้น
            #    - ไม่งั้น fallback เดิม jobs/output/<job_id>/result.mp4
            try:
                best_key = _pick_best_output_key(selected_job_id)

                # พยายามดาวน์โหลดเป็น bytes (เหมือนเดิม) ก่อน
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=best_key)
                data = obj["Body"].read()

            except ClientError as ce:
                code = ce.response.get("Error", {}).get("Code")
                if code == "NoSuchKey":
                    st.error("Result not ready yet. Please wait and refresh.")
                else:
                    st.error(f"Download error: {ce}")

                # fallback: แสดงลิงก์ presigned ให้กดดาวน์โหลดได้ (เพิ่มเฉพาะตอน error)
                try:
                    best_key = _pick_best_output_key(selected_job_id)
                    if s3_key_exists(best_key):
                        url = presigned_get_url(best_key, expires=3600)
                        if hasattr(st, "link_button"):
                            st.link_button("Download result.mp4", url)
                        else:
                            st.markdown(f"[Download result.mp4]({url})")
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Download error: {e}")

            else:
                # ตั้งชื่อไฟล์ให้เหมือนเดิม แต่ถ้าจริง ๆ เป็น overlay/dots ก็ยังดาวน์โหลดได้
                st.download_button(
                    "Download result.mp4",
                    data=data,
                    file_name=f"{selected_job_id}_result.mp4",
                    mime="video/mp4",
                )
