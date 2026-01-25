# 2_Submit_Job.py — Video Analysis (วิเคราะห์วิดีโอ)
# Upload once -> get 4 downloads:
#   1) Dots video
#   2) Skeleton video
#   3) English report
#   4) Thai report
#
# IMPORTANT:
# - This page uses the LEGACY queue that your worker already supports:
#     jobs/pending/<job_id>.json
#     jobs/pending/<job_id>/input/input.mp4
#   Output expected:
#     jobs/output/<job_id>/result.mp4
#
# - Reports are expected to be written somewhere in S3 by your report generator.
#   This page just "looks for" report files and creates download links if found.

import os
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import streamlit as st
import boto3
from botocore.exceptions import ClientError


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Video Analysis (วิเคราะห์วิดีโอ)", layout="wide")

# -------------------------
# Env / S3
# -------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_OUTPUT_PREFIX = "jobs/output/"

# -------------------------
# Report keys (customize if your report generator uses different paths)
# We'll try these candidates (first found wins).
# You can adjust without touching the rest of the logic.
# -------------------------
def report_candidates(group_id: str) -> Dict[str, List[str]]:
    """
    Return candidate S3 keys for EN/TH reports.
    Adjust these paths to match where your system actually writes reports.
    """
    return {
        "th": [
            f"jobs/output/{group_id}/report_th.pdf",
            f"jobs/output/{group_id}/thai_report.pdf",
            f"jobs/output/{group_id}/report_th.docx",
            f"jobs/output/{group_id}/thai_report.docx",
            f"jobs/{group_id}/output/report_th.pdf",
            f"jobs/{group_id}/output/report_th.docx",
        ],
        "en": [
            f"jobs/output/{group_id}/report_en.pdf",
            f"jobs/output/{group_id}/english_report.pdf",
            f"jobs/output/{group_id}/report_en.docx",
            f"jobs/output/{group_id}/english_report.docx",
            f"jobs/{group_id}/output/report_en.pdf",
            f"jobs/{group_id}/output/report_en.docx",
        ],
    }


# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def guess_content_type(filename: str) -> str:
    fn = (filename or "").lower()
    if fn.endswith(".mp4"):
        return "video/mp4"
    if fn.endswith(".mov"):
        return "video/quicktime"
    if fn.endswith(".m4v"):
        return "video/x-m4v"
    if fn.endswith(".webm"):
        return "video/webm"
    if fn.endswith(".pdf"):
        return "application/pdf"
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return "application/octet-stream"


def s3_put_bytes(key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def presigned_get_url(key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
    """
    Force download with ResponseContentDisposition
    """
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        params["ResponseContentType"] = guess_content_type(filename)

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )


def create_legacy_job(
    *,
    file_bytes: bytes,
    mode: str,
    dot_radius: int,
    skeleton_color: str,
    skeleton_thickness: int,
    note: str,
) -> Dict[str, Any]:
    """
    LEGACY worker contract:
      input:  jobs/pending/<job_id>/input/input.mp4
      job:    jobs/pending/<job_id>.json
      output: jobs/output/<job_id>/result.mp4
    """
    job_id = new_job_id()

    input_key = f"{JOBS_PENDING_PREFIX}{job_id}/input/input.mp4"
    output_key = f"{JOBS_OUTPUT_PREFIX}{job_id}/result.mp4"

    s3_put_bytes(input_key, file_bytes, content_type="video/mp4")

    now = utc_now_iso()
    job: Dict[str, Any] = {
        "job_id": job_id,
        "status": "pending",
        "mode": mode,  # "dots" or "skeleton"
        "input_key": input_key,
        "output_key": output_key,
        "created_at": now,
        "updated_at": now,
        "error": None,
        "note": note or "",
    }

    # mode-specific fields (match your working app.py)
    if mode == "dots":
        job["dot_radius"] = int(dot_radius)
    elif mode == "skeleton":
        job["skeleton_line_color"] = str(skeleton_color)
        job["skeleton_line_thickness"] = int(skeleton_thickness)

    job_json_key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    s3_put_json(job_json_key, job)

    return job


def find_first_existing_key(candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k and s3_key_exists(k):
            return k
    return None


# -------------------------
# UI (match your example layout)
# -------------------------
st.markdown("# Video Analysis (วิเคราะห์วิดีโอ)")
st.markdown("Upload your video, then click **Analysis**. (อัปโหลดวิดีโอ แล้วกด Analysis)")

st.write("")  # spacer

uploaded = st.file_uploader(
    "Video (MP4) (วิดีโอ MP4)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
)

# Keep the page simple like your example, but we still need the params:
# (use same defaults as your working app.py)
DEFAULT_DOT_RADIUS = 5
DEFAULT_SKELETON_COLOR = "#00FF00"
DEFAULT_SKELETON_THICKNESS = 2

# We keep note optional (hidden-ish but still there)
note = st.text_input("Note (optional)", value="", label_visibility="collapsed", placeholder="")

btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    analyze_clicked = st.button("Analysis (วิเคราะห์)", type="primary", disabled=(uploaded is None))
with btn_col2:
    reset_clicked = st.button("Reset (รีเซ็ต)")

if reset_clicked:
    # wipe only this page state
    for k in [
        "group_id",
        "dots_job_id",
        "skeleton_job_id",
        "submitted_at",
        "last_status",
        "last_message",
    ]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

# Submit on Analysis
if analyze_clicked and uploaded is not None:
    try:
        video_bytes = uploaded.getvalue()

        # Create a group id (so report keys can be grouped even though dots/skeleton have their own job_id)
        group_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        st.session_state["group_id"] = group_id

        # Create 2 legacy jobs: dots + skeleton
        job_dots = create_legacy_job(
            file_bytes=video_bytes,
            mode="dots",
            dot_radius=DEFAULT_DOT_RADIUS,
            skeleton_color=DEFAULT_SKELETON_COLOR,
            skeleton_thickness=DEFAULT_SKELETON_THICKNESS,
            note=note,
        )
        job_skeleton = create_legacy_job(
            file_bytes=video_bytes,
            mode="skeleton",
            dot_radius=DEFAULT_DOT_RADIUS,
            skeleton_color=DEFAULT_SKELETON_COLOR,
            skeleton_thickness=DEFAULT_SKELETON_THICKNESS,
            note=note,
        )

        st.session_state["dots_job_id"] = job_dots["job_id"]
        st.session_state["skeleton_job_id"] = job_skeleton["job_id"]
        st.session_state["submitted_at"] = utc_now_iso()

        st.session_state["last_status"] = "processing"
        st.session_state["last_message"] = "processing video (กำลังประมวลผลวิดีโอ)"

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)

# Show processing/done + Downloads (same vibe as your example)
group_id = st.session_state.get("group_id")
dots_job_id = st.session_state.get("dots_job_id")
skeleton_job_id = st.session_state.get("skeleton_job_id")

if group_id and dots_job_id and skeleton_job_id:
    st.write("")
    st.write(st.session_state.get("last_message", "processing video (กำลังประมวลผลวิดีโอ)"))

    # Check readiness (videos)
    dots_out_key = f"{JOBS_OUTPUT_PREFIX}{dots_job_id}/result.mp4"
    skel_out_key = f"{JOBS_OUTPUT_PREFIX}{skeleton_job_id}/result.mp4"

    dots_ready = s3_key_exists(dots_out_key)
    skel_ready = s3_key_exists(skel_out_key)

    # Reports (optional) — find if they exist
    rc = report_candidates(group_id)
    report_th_key = find_first_existing_key(rc["th"])
    report_en_key = find_first_existing_key(rc["en"])

    all_ready = dots_ready and skel_ready and (report_th_key is not None) and (report_en_key is not None)

    # Auto-refresh a little (without being too aggressive)
    # If not ready, give a "check again" button like your “check output” flow.
    check_col1, check_col2 = st.columns([1, 1])
    with check_col1:
        check_now = st.button("Check output (ตรวจผลอีกครั้ง)")
    with check_col2:
        st.caption(f"Group ID: `{group_id}`")

    if check_now:
        st.rerun()

    if all_ready:
        st.success("Done. Download your files below. (เสร็จแล้ว ดาวน์โหลดไฟล์ได้ด้านล่าง)")
        st.session_state["last_message"] = "Done. Download your files below. (เสร็จแล้ว ดาวน์โหลดไฟล์ได้ด้านล่าง)"
    else:
        st.info("processing video (กำลังประมวลผลวิดีโอ)")

    st.markdown("## Downloads (ดาวน์โหลด)")

    d1, d2 = st.columns(2)

    # 1) Dots video
    with d1:
        if dots_ready:
            url = presigned_get_url(dots_out_key, filename=f"{dots_job_id}_dots.mp4")
            if hasattr(st, "link_button"):
                st.link_button("Download: Processed VDO for dots (วิดีโอประมวลผลสำหรับจุด)", url)
            else:
                st.markdown(f"[Download: Processed VDO for dots (วิดีโอประมวลผลสำหรับจุด)]({url})")
        else:
            st.button(
                "Download: Processed VDO for dots (วิดีโอประมวลผลสำหรับจุด)",
                disabled=True,
            )

    # 2) Skeleton video
    with d2:
        if skel_ready:
            url = presigned_get_url(skel_out_key, filename=f"{skeleton_job_id}_skeleton.mp4")
            if hasattr(st, "link_button"):
                st.link_button("Download: Processed VDO for skeleton (วิดีโอประมวลผลสำหรับโครงกระดูก)", url)
            else:
                st.markdown(f"[Download: Processed VDO for skeleton (วิดีโอประมวลผลสำหรับโครงกระดูก)]({url})")
        else:
            st.button(
                "Download: Processed VDO for skeleton (วิดีโอประมวลผลสำหรับโครงกระดูก)",
                disabled=True,
            )

    d3, d4 = st.columns(2)

    # 3) Thai report
    with d3:
        if report_th_key:
            fname = report_th_key.split("/")[-1] or "thai_report.pdf"
            url = presigned_get_url(report_th_key, filename=fname)
            if hasattr(st, "link_button"):
                st.link_button("Download: Thai Report (รายงานภาษาไทย)", url)
            else:
                st.markdown(f"[Download: Thai Report (รายงานภาษาไทย)]({url})")
        else:
            st.button("Download: Thai Report (รายงานภาษาไทย)", disabled=True)

    # 4) English report
    with d4:
        if report_en_key:
            fname = report_en_key.split("/")[-1] or "english_report.pdf"
            url = presigned_get_url(report_en_key, filename=fname)
            if hasattr(st, "link_button"):
                st.link_button("Download: English Report (รายงานภาษาอังกฤษ)", url)
            else:
                st.markdown(f"[Download: English Report (รายงานภาษาอังกฤษ)]({url})")
        else:
            st.button("Download: English Report (รายงานภาษาอังกฤษ)", disabled=True)

    st.write("")
    st.caption(
        "หมายเหตุ: หน้านี้ intentionally ใช้ legacy queue (jobs/pending/*.json) เพื่อให้ dots/skeleton กลับมาทำงานเหมือนเดิม "
        "และจะโชว์ปุ่ม report ก็ต่อเมื่อไฟล์ report ถูกเขียนลง S3 ตาม path ที่กำหนดไว้ในโค้ด"
    )

else:
    # initial empty state — keep simple like your example
    st.write("")
