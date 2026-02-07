# pages/2_Submit_Job.py — Video Analysis (วิเคราะห์วิดีโอ)
# ------------------------------------------------------------
# Upload once (shared key) -> get downloads:
#   1) Dots video (MP4)
#   2) Skeleton video (MP4)
#   3) English report (DOCX)
#   4) Thai report (DOCX)
#
# ✅ DOCX only (NO PDF)
# ✅ LEGACY queue:
#     jobs/pending/<job_id>.json
# ✅ Self-healing submit:
#     - Writes manifest to S3 for the group
#     - Enqueues 4 jobs with verification
#     - Auto-repairs missing jobs from manifest on refresh/page load
# ------------------------------------------------------------

import os
import json
import uuid
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import streamlit as st
import boto3


# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Video Analysis (วิเคราะห์วิดีโอ)", layout="wide")


# -------------------------
# Env / S3
# -------------------------
# Keep compatibility with your current Render env names.
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"

JOBS_OUTPUT_PREFIX = "jobs/output/"
JOBS_GROUP_PREFIX = "jobs/groups/"

MANIFEST_REL = "meta/manifest.json"  # stored under jobs/groups/<group_id>/meta/manifest.json


# -------------------------
# Helpers
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def new_group_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}"


def safe_slug(text: str, fallback: str = "user") -> str:
    t = (text or "").strip()
    if not t:
        return fallback
    out: List[str] = []
    for ch in t:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        elif ch.isspace():
            out.append("_")
    s = "".join(out).strip("_")
    return s if s else fallback


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
    if fn.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if fn.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def s3_put_bytes(key: str, data: bytes, content_type: str) -> None:
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_put_json(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


def s3_put_json_pretty(key: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json; charset=utf-8",
    )


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def s3_read_json(key: str) -> Optional[Dict[str, Any]]:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        raw = obj["Body"].read().decode("utf-8")
        return json.loads(raw)
    except Exception:
        return None


def presigned_get_url(key: str, expires: int = 3600, filename: Optional[str] = None) -> str:
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}
    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        params["ResponseContentType"] = guess_content_type(filename)

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )


def manifest_key_for_group(group_id: str) -> str:
    return f"{JOBS_GROUP_PREFIX}{group_id}/{MANIFEST_REL}"


def write_manifest(group_id: str, manifest: Dict[str, Any]) -> str:
    k = manifest_key_for_group(group_id)
    s3_put_json_pretty(k, manifest)
    return k


def read_manifest(group_id: str) -> Optional[Dict[str, Any]]:
    return s3_read_json(manifest_key_for_group(group_id))


def enqueue_with_verify(job: Dict[str, Any], retries: int = 5, delay_s: float = 0.25) -> str:
    """
    Put job json -> verify head_object -> retry.
    Guarantees the job json exists in S3 when returned (or raises).
    """
    job_id = str(job["job_id"])
    key = f"{JOBS_PENDING_PREFIX}{job_id}.json"
    last_err: Optional[Exception] = None

    for _ in range(retries):
        try:
            s3_put_json(key, job)
            if s3_key_exists(key):
                return key
        except Exception as e:
            last_err = e
        time.sleep(delay_s)

    raise RuntimeError(f"enqueue_with_verify failed for {job_id}: {last_err}")


def find_job_json(job_id: str) -> Optional[str]:
    """
    Return the key path of the job json if found in any known prefix.
    """
    candidates = [
        f"{JOBS_PENDING_PREFIX}{job_id}.json",
        f"{JOBS_PROCESSING_PREFIX}{job_id}.json",
        f"{JOBS_FINISHED_PREFIX}{job_id}.json",
        f"{JOBS_FAILED_PREFIX}{job_id}.json",
    ]
    for k in candidates:
        if s3_key_exists(k):
            return k
    return None


def infer_job_bucket_status(job_key: str) -> str:
    if job_key.startswith(JOBS_PENDING_PREFIX):
        return "pending"
    if job_key.startswith(JOBS_PROCESSING_PREFIX):
        return "processing"
    if job_key.startswith(JOBS_FINISHED_PREFIX):
        return "finished"
    if job_key.startswith(JOBS_FAILED_PREFIX):
        return "failed"
    return "unknown"


def job_exists_anywhere(job_id: str) -> bool:
    return find_job_json(job_id) is not None


def repair_from_manifest(group_id: str) -> Dict[str, Any]:
    """
    If any job described in manifest is missing from all buckets,
    re-enqueue it to pending. Returns repair report.
    """
    mf = read_manifest(group_id) or {}
    jobs_obj = mf.get("jobs") or {}
    repaired: List[Dict[str, Any]] = []
    skipped: List[str] = []
    errors: List[Dict[str, Any]] = []

    for name, job in jobs_obj.items():
        try:
            jid = str(job.get("job_id") or "")
            if not jid:
                continue

            if job_exists_anywhere(jid):
                skipped.append(name)
                continue

            k = enqueue_with_verify(job)
            repaired.append({"name": name, "job_id": jid, "key": k})

        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    return {"repaired": repaired, "skipped": skipped, "errors": errors}


def build_output_keys(group_id: str) -> Dict[str, str]:
    """
    Keep outputs under jobs/output/groups/<group_id>/...
    """
    base = f"{JOBS_OUTPUT_PREFIX}groups/{group_id}/"
    return {
        "dots_video": base + "dots.mp4",
        "skeleton_video": base + "skeleton.mp4",
        "report_en_docx": base + "report_en.docx",
        "report_th_docx": base + "report_th.docx",
        "debug_en": base + "debug_en.json",
        "debug_th": base + "debug_th.json",
    }


def ensure_session_defaults() -> None:
    if "last_group_id" not in st.session_state:
        st.session_state["last_group_id"] = None
    if "last_outputs" not in st.session_state:
        st.session_state["last_outputs"] = None
    if "last_jobs" not in st.session_state:
        st.session_state["last_jobs"] = None
    if "last_job_json_keys" not in st.session_state:
        st.session_state["last_job_json_keys"] = None
    if "manifest_key" not in st.session_state:
        st.session_state["manifest_key"] = None


# -------------------------
# UI
# -------------------------
ensure_session_defaults()

st.markdown("# Video Analysis (วิเคราะห์วิดีโอ)")
st.caption("Upload once, then click **Video Analysis** to generate dots + skeleton + reports (EN/TH). ✅ DOCX only (NO PDF)")
st.caption(f"Using S3 bucket: `{AWS_BUCKET}` | region: `{AWS_REGION}`")

with st.expander("Optional: User Name (ชื่อผู้ใช้) — ใช้สำหรับติดตามงาน", expanded=False):
    user_name = st.text_input("Enter User Name", value="", placeholder="e.g., Rung / Founder / Co-Founder")
    st.caption("Tip: ไม่ใส่ก็ได้ ระบบจะสร้าง group id ให้อัตโนมัติ")

uploaded = st.file_uploader(
    "Video (MP4/MOV/M4V/WEBM)",
    type=["mp4", "mov", "m4v", "webm"],
    accept_multiple_files=False,
)

colA, colB = st.columns([1, 1])
with colA:
    run = st.button("🎬 Video Analysis", type="primary", use_container_width=True)
with colB:
    refresh = st.button("🔄 Refresh", use_container_width=True)

note = st.empty()

if refresh:
    st.rerun()


# -------------------------
# Submit jobs (Self-healing)
# -------------------------
if run:
    if not uploaded:
        note.error("Please upload a video first.")
        st.stop()

    base_user = safe_slug(user_name, fallback="user")
    group_id = f"{new_group_id()}__{base_user}"
    input_key = f"{JOBS_GROUP_PREFIX}{group_id}/input/input.mp4"

    # Upload shared input
    try:
        s3_put_bytes(
            key=input_key,
            data=uploaded.getvalue(),
            content_type=guess_content_type(uploaded.name or "input.mp4"),
        )
    except Exception as e:
        note.error(f"Upload to S3 failed: {e}")
        st.stop()

    outputs = build_output_keys(group_id)
    created_at = utc_now_iso()

    # 1) dots job
    job_dots = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "dots",
        "input_key": input_key,
        "output_key": outputs["dots_video"],
        "user_name": user_name or "",
    }

    # 2) skeleton job
    job_skel = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "skeleton",
        "input_key": input_key,
        "output_key": outputs["skeleton_video"],
        "user_name": user_name or "",
    }

    # 3) report EN job (DOCX only)
    job_rep_en = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "report",
        "language": "en",
        "input_key": input_key,
        "output_docx_key": outputs["report_en_docx"],
        "output_debug_key": outputs["debug_en"],
        "user_name": user_name or "",
        "include_first_impression": True,
    }

    # 4) report TH job (DOCX only)
    job_rep_th = {
        "job_id": new_job_id(),
        "group_id": group_id,
        "created_at": created_at,
        "status": "pending",
        "mode": "report",
        "language": "th",
        "input_key": input_key,
        "output_docx_key": outputs["report_th_docx"],
        "output_debug_key": outputs["debug_th"],
        "user_name": user_name or "",
        "include_first_impression": True,
    }

    # Manifest: single source of truth for this group
    manifest = {
        "group_id": group_id,
        "created_at": created_at,
        "bucket": AWS_BUCKET,
        "region": AWS_REGION,
        "input_key": input_key,
        "outputs": outputs,
        "jobs": {
            "dots": job_dots,
            "skeleton": job_skel,
            "report_en": job_rep_en,
            "report_th": job_rep_th,
        },
        "state": "created",
    }

    try:
        manifest_s3_key = write_manifest(group_id, manifest)

        # Enqueue with verification (never silently missing)
        k1 = enqueue_with_verify(job_dots)
        k2 = enqueue_with_verify(job_skel)
        k3 = enqueue_with_verify(job_rep_en)
        k4 = enqueue_with_verify(job_rep_th)

        manifest["state"] = "enqueued"
        manifest["enqueued_at"] = utc_now_iso()
        manifest["job_json_keys"] = {
            "dots": k1,
            "skeleton": k2,
            "report_en": k3,
            "report_th": k4,
        }
        write_manifest(group_id, manifest)

    except Exception as e:
        # Keep manifest for repair, even if enqueue failed mid-way
        manifest["state"] = "enqueue_error"
        manifest["error"] = str(e)
        try:
            write_manifest(group_id, manifest)
        except Exception:
            pass
        note.error(f"Enqueue job failed: {e}")
        st.stop()

    st.session_state["last_group_id"] = group_id
    st.session_state["last_outputs"] = outputs
    st.session_state["last_jobs"] = {
        "dots": job_dots["job_id"],
        "skeleton": job_skel["job_id"],
        "report_en": job_rep_en["job_id"],
        "report_th": job_rep_th["job_id"],
    }
    st.session_state["last_job_json_keys"] = {
        "dots": k1,
        "skeleton": k2,
        "report_en": k3,
        "report_th": k4,
    }
    st.session_state["manifest_key"] = manifest_s3_key

    note.success(f"Submitted! group_id = {group_id}")
    st.info("Wait a bit, then press Refresh. (We will show job status + outputs.)")


# -------------------------
# Download section
# -------------------------
group_id = st.session_state.get("last_group_id")
outputs = st.session_state.get("last_outputs") or {}
jobs = st.session_state.get("last_jobs") or {}
job_json_keys = st.session_state.get("last_job_json_keys") or {}
manifest_key = st.session_state.get("manifest_key")

st.divider()
st.subheader("Downloads (ผลลัพธ์สำหรับดาวน์โหลด)")

if not group_id:
    st.caption("No job submitted yet. Upload a video and click **Video Analysis**.")
    st.stop()

st.caption(f"Group: `{group_id}`")

# Auto-repair: ensures all 4 jobs exist even if submit got interrupted
rep = repair_from_manifest(group_id)
if rep.get("repaired"):
    st.info(f"🛠️ Auto-repaired missing jobs: {[x['name'] for x in rep['repaired']]}")
if rep.get("errors"):
    st.warning(f"Repair errors: {rep['errors']}")


def download_block(title: str, key: str, filename: str) -> None:
    if not key:
        st.write(f"- {title}: (missing key)")
        return
    ready = s3_key_exists(key)
    if ready:
        url = presigned_get_url(key, expires=3600, filename=filename)
        st.success(f"✅ {title} ready")
        st.link_button(f"Download {title}", url, use_container_width=True)
        st.code(key, language="text")
    else:
        st.warning(f"⏳ {title} not ready yet")
        st.code(key, language="text")


def job_status_block(label: str, job_id: str) -> None:
    if not job_id:
        st.write(f"- {label}: (missing job_id)")
        return

    found_key = find_job_json(job_id)
    if not found_key:
        st.error(f"❌ {label}: job json not found anywhere (pending/processing/finished/failed)")
        st.code(f"{JOBS_PENDING_PREFIX}{job_id}.json", language="text")
        return

    bucket_status = infer_job_bucket_status(found_key)
    payload = s3_read_json(found_key) or {}
    inner_status = (payload.get("status") or "").strip() or "—"
    msg = (payload.get("message") or "").strip()

    if bucket_status == "failed":
        st.error(f"🧨 {label}: {bucket_status} (job.status={inner_status})")
    elif bucket_status == "finished":
        st.success(f"✅ {label}: {bucket_status} (job.status={inner_status})")
    elif bucket_status == "processing":
        st.info(f"🟦 {label}: {bucket_status} (job.status={inner_status})")
    else:
        st.warning(f"⏳ {label}: {bucket_status} (job.status={inner_status})")

    st.code(found_key, language="text")
    if msg:
        st.write(f"**message:** {msg}")


# --- Job status inspector ---
with st.expander("🔎 Job Status Inspector (ดูสถานะงานจาก S3 JSON)", expanded=True):
    st.markdown("ถ้า Report ไม่ออก ให้ดูตรงนี้ก่อนว่า job ไปค้าง/ล้มตรงไหน")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**Video jobs**")
        job_status_block("Dots", jobs.get("dots", ""))
        job_status_block("Skeleton", jobs.get("skeleton", ""))
    with cB:
        st.markdown("**Report jobs (DOCX only)**")
        job_status_block("Report EN", jobs.get("report_en", ""))
        job_status_block("Report TH", jobs.get("report_th", ""))

    if manifest_key:
        st.markdown("**Manifest key (source of truth)**")
        st.code(manifest_key, language="text")
        mf = s3_read_json(manifest_key) or {}
        if mf:
            st.json({"state": mf.get("state"), "created_at": mf.get("created_at"), "enqueued_at": mf.get("enqueued_at")})

    if job_json_keys:
        st.markdown("**Enqueued JSON keys (ตอน submit ล่าสุด)**")
        st.json(job_json_keys)

    st.markdown("**Debug JSON outputs (ถ้ามี)**")
    dbg1 = outputs.get("debug_en", "")
    dbg2 = outputs.get("debug_th", "")
    if dbg1:
        if s3_key_exists(dbg1):
            st.success("✅ debug_en.json ready")
            st.code(dbg1, language="text")
        else:
            st.warning("⏳ debug_en.json not ready yet")
            st.code(dbg1, language="text")
    if dbg2:
        if s3_key_exists(dbg2):
            st.success("✅ debug_th.json ready")
            st.code(dbg2, language="text")
        else:
            st.warning("⏳ debug_th.json not ready yet")
            st.code(dbg2, language="text")


# --- Downloads ---
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Videos")
    download_block("Dots video", outputs.get("dots_video", ""), "dots.mp4")
    download_block("Skeleton video", outputs.get("skeleton_video", ""), "skeleton.mp4")

with c2:
    st.markdown("### Reports (DOCX only)")
    st.markdown("**English**")
    download_block("Report EN (DOCX)", outputs.get("report_en_docx", ""), "report_en.docx")

    st.markdown("**Thai**")
    download_block("Report TH (DOCX)", outputs.get("report_th_docx", ""), "report_th.docx")

st.caption("Tip: ถ้าไฟล์ยังไม่ขึ้น ให้กด Refresh หรือรอสักครู่แล้วลองใหม่")
