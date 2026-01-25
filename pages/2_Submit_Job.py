# 2_Submit_Job.py — Submit Job to S3 (NEW + LEGACY pending bridge)
# - NEW: jobs/<job_id>/job.json + jobs/<job_id>/status.json
# - LEGACY: jobs/pending/<job_id>.json  (so ai-people-reader-worker can see jobs)
# - Download links persist (no disappearing after click)

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import streamlit as st
import boto3
from botocore.exceptions import ClientError

# OPTIONAL: call Presentation Analysis API (if requests not installed, it won't break)
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


# =========================
# Page setup
# =========================
st.set_page_config(page_title="Submit Job (S3)", layout="wide")
st.title("🚀 Submit Job to S3 (Safe / Separate Page)")

st.caption(
    "หน้านี้เป็นหน้าใหม่แยกจาก app.py เดิม: ทำแค่อัปโหลด + สร้าง job ใน S3 "
    "และทำ 'legacy pending bridge' เพื่อให้ ai-people-reader-worker เดิมเห็นงานทันที"
)

# =========================
# Env
# =========================
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

with st.expander("🔧 Environment (read-only)", expanded=False):
    st.write("AWS_BUCKET =", AWS_BUCKET)
    st.write("AWS_REGION =", AWS_REGION)

if not AWS_BUCKET:
    st.error("Missing AWS_BUCKET (or S3_BUCKET) environment variable in Render.")
    st.stop()

s3 = boto3.client("s3", region_name=AWS_REGION)


# =========================
# Helpers
# =========================
def new_job_id() -> str:
    # ให้เหมือน log เดิมที่เจอ ...__xxxx
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def s3_put_json(key: str, obj: dict):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(obj, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json",
    )


def s3_put_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def s3_get_json(key: str) -> dict | None:
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        data = obj["Body"].read().decode("utf-8", errors="replace")
        return json.loads(data)
    except Exception:
        return None


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


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
    if fn.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


def presigned_get_url(
    key: str,
    expires: int = 3600,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """
    Force download by setting Content-Disposition attachment.
    """
    params: dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}

    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'

    if content_type:
        params["ResponseContentType"] = content_type

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )


# =========================
# ✅ Presentation Analysis integration helpers
# =========================
def normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    return u[:-1] if u.endswith("/") else u


def build_pa_ui_url(pa_base_url: str, job_id: str, lang: str) -> str:
    base = normalize_base_url(pa_base_url)
    return f"{base}/?job_id={job_id}&lang={lang}"


def try_generate_report_via_pa_api(pa_base_url: str, job_id: str, lang: str) -> dict | None:
    """
    Try to call PA API to generate report and return response containing S3 key.
    If no requests or endpoint not available, returns None.
    """
    if requests is None:
        return None

    base = normalize_base_url(pa_base_url)
    endpoint = f"{base}/api/generate_report"  # adjust if your PA uses different endpoint

    try:
        r = requests.get(endpoint, params={"job_id": job_id, "lang": lang}, timeout=60)
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/json" not in ctype:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def extract_report_s3_key(api_response: dict) -> str | None:
    for k in ["report_key", "s3_key", "output_key", "key"]:
        v = api_response.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    outputs = api_response.get("outputs")
    if isinstance(outputs, dict):
        v = outputs.get("report")
        if isinstance(v, str) and v.strip():
            return v.strip()

    return None


# =========================
# ✅ Job schema builders
# =========================
def build_job_manifest_new(job_id: str, input_key: str, modes: list[str], note: str = "") -> dict:
    # NEW schema (your page uses)
    return {
        "job_id": job_id,
        "input_key": input_key,
        "modes": modes,
        "note": note,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "submit-v2-new+legacy",
    }


def build_job_manifest_legacy(job_id: str, input_key: str, modes: list[str], note: str = "") -> dict:
    """
    LEGACY schema (safe & compatible):
    We include several field names so old worker won't miss it.
    Worker log shows it scans: jobs/pending/<job_id>.json
    """
    # many legacy workers use "mode" (single) not "modes"
    # choose first mode if exists
    mode_single = modes[0] if modes else "overlay"

    return {
        "job_id": job_id,
        "input_key": input_key,            # used by newer workers
        "video_key": input_key,            # used by some older workers
        "mode": mode_single,               # single mode fallback
        "modes": modes,                    # keep list too
        "note": note,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "legacy-bridge-v1",
    }


# =========================
# UI: Submit
# =========================
st.subheader("1) Upload video + create job")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "webm"])
    note = st.text_input("Note (optional)", value="")

with col2:
    st.markdown("### Modes to request")
    mode_overlay = st.checkbox("overlay", value=False)
    mode_dots = st.checkbox("dots", value=True)
    mode_skeleton = st.checkbox("skeleton", value=False)
    mode_report = st.checkbox("report", value=False)

modes: list[str] = []
if mode_overlay:
    modes.append("overlay")
if mode_dots:
    modes.append("dots")
if mode_skeleton:
    modes.append("skeleton")
if mode_report:
    modes.append("report")

if not modes:
    st.warning("Please select at least 1 mode (overlay/dots/skeleton/report).")

st.caption("✅ IMPORTANT: เพื่อให้ worker เดิมเห็นงาน เราจะเขียน LEGACY pending ที่ jobs/pending/<job_id>.json ด้วย")


# Toggle: legacy bridge (default ON)
legacy_bridge = st.checkbox("✅ Legacy bridge (write jobs/pending/<job_id>.json for existing worker)", value=True)


if st.button("🚀 Submit job", disabled=(uploaded is None or len(modes) == 0)):
    try:
        job_id = new_job_id()
        filename = uploaded.name if uploaded else "input.mp4"
        content_type = guess_content_type(filename)

        # Upload input video (keep inside jobs/<job_id>/input/..)
        input_key = f"jobs/{job_id}/input/{filename}"
        video_bytes = uploaded.getvalue()
        s3_put_bytes(input_key, video_bytes, content_type=content_type)

        # NEW manifest (for this UI)
        job_new = build_job_manifest_new(job_id, input_key, modes=modes, note=note)
        s3_put_json(f"jobs/{job_id}/job.json", job_new)
        s3_put_json(f"jobs/{job_id}/status.json", {"status": "queued", "job_id": job_id})

        # LEGACY pending (so ai-people-reader-worker will pick it up)
        if legacy_bridge:
            job_legacy = build_job_manifest_legacy(job_id, input_key, modes=modes, note=note)
            pending_key = f"jobs/pending/{job_id}.json"
            s3_put_json(pending_key, job_legacy)

        st.session_state["last_job_id"] = job_id

        st.success("Submitted ✅")
        st.write("Job ID:", job_id)
        st.code(json.dumps(job_new, ensure_ascii=False, indent=2))

        if legacy_bridge:
            st.info(f"Legacy pending written: jobs/pending/{job_id}.json  (worker เดิมจะหยิบงานจากตรงนี้)")
        st.info(f"New job folder: jobs/{job_id}/...")

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)


# =========================
# Verify / Downloads
# =========================
st.divider()
st.subheader("2) Verify job exists (read-only)")

job_id_check = st.text_input("Job ID to check", value=st.session_state.get("last_job_id", ""))

pa_default = os.getenv("PRESENTATION_ANALYSIS_URL", "https://presentation-analysis.onrender.com")
PA_BASE_URL = st.text_input("Presentation Analysis URL (for report)", value=pa_default)


def find_status_object(job_id: str) -> tuple[str | None, dict | None]:
    """
    Try multiple possible locations for status.json.
    Returns (status_key, status_obj)
    """
    candidates = [
        f"jobs/{job_id}/status.json",              # NEW
        f"jobs/output/{job_id}/status.json",       # common LEGACY
        f"jobs/{job_id}/output/status.json",       # another common pattern
        f"jobs/{job_id}/status/status.json",       # rare but safe
    ]
    for k in candidates:
        if s3_key_exists(k):
            return k, s3_get_json(k)
    return None, None


def remember_url(slot: str, job_id: str, url: str):
    st.session_state.setdefault("download_urls", {})
    st.session_state["download_urls"].setdefault(job_id, {})
    st.session_state["download_urls"][job_id][slot] = url


def get_remembered_url(slot: str, job_id: str) -> str | None:
    return (
        st.session_state.get("download_urls", {})
        .get(job_id, {})
        .get(slot)
    )


if st.button("Check status.json"):
    if not job_id_check.strip():
        st.warning("Please enter job_id")
    else:
        jid = job_id_check.strip()

        status_key, status_obj = find_status_object(jid)

        if not status_key or not isinstance(status_obj, dict):
            st.error("Cannot find status.json for this job in known locations.")
            st.write("Tried:", [
                f"jobs/{jid}/status.json",
                f"jobs/output/{jid}/status.json",
                f"jobs/{jid}/output/status.json",
                f"jobs/{jid}/status/status.json",
            ])
        else:
            st.success(f"Found status: {status_key}")
            st.json(status_obj)

            outputs = (status_obj or {}).get("outputs") or {}
            if not isinstance(outputs, dict) or len(outputs) == 0:
                st.info("ยังไม่มี outputs ใน status.json (รอ worker เขียน outputs ก่อน)")
            else:
                st.subheader("3) Downloads")

                # --- Render download blocks for each output ---
                for name, out_key in outputs.items():
                    if not isinstance(out_key, str) or not out_key.strip():
                        continue

                    out_key = out_key.strip()
                    name_lc = str(name).lower().strip()

                    # ------------------
                    # REPORT
                    # ------------------
                    if name_lc == "report":
                        st.markdown("#### 📄 Report (from Presentation Analysis)")

                        col_th, col_en = st.columns(2)

                        with col_th:
                            if st.button("⬇️ Prepare report link (TH)", key=f"btn_prepare_report_th_{jid}"):
                                pa = normalize_base_url(PA_BASE_URL)
                                api_res = try_generate_report_via_pa_api(pa, jid, "th")
                                report_key = extract_report_s3_key(api_res) if isinstance(api_res, dict) else None

                                if report_key and s3_key_exists(report_key):
                                    fname = report_key.split("/")[-1] or "report_th.pdf"
                                    url = presigned_get_url(
                                        report_key,
                                        expires=3600,
                                        filename=fname,
                                        content_type=guess_content_type(fname),
                                    )
                                    remember_url("report_th", jid, url)
                                else:
                                    # fallback UI URL (still remember so it won't disappear)
                                    remember_url("report_th", jid, build_pa_ui_url(pa, jid, "th"))

                            url_th = get_remembered_url("report_th", jid)
                            if url_th:
                                if "presentation-analysis" in url_th and "/?job_id=" in url_th:
                                    st.link_button("Open Presentation Analysis (TH)", url_th)
                                else:
                                    st.link_button("Download report (TH) — file", url_th)

                        with col_en:
                            if st.button("⬇️ Prepare report link (EN)", key=f"btn_prepare_report_en_{jid}"):
                                pa = normalize_base_url(PA_BASE_URL)
                                api_res = try_generate_report_via_pa_api(pa, jid, "en")
                                report_key = extract_report_s3_key(api_res) if isinstance(api_res, dict) else None

                                if report_key and s3_key_exists(report_key):
                                    fname = report_key.split("/")[-1] or "report_en.pdf"
                                    url = presigned_get_url(
                                        report_key,
                                        expires=3600,
                                        filename=fname,
                                        content_type=guess_content_type(fname),
                                    )
                                    remember_url("report_en", jid, url)
                                else:
                                    remember_url("report_en", jid, build_pa_ui_url(pa, jid, "en"))

                            url_en = get_remembered_url("report_en", jid)
                            if url_en:
                                if "presentation-analysis" in url_en and "/?job_id=" in url_en:
                                    st.link_button("Open Presentation Analysis (EN)", url_en)
                                else:
                                    st.link_button("Download report (EN) — file", url_en)

                        # skip default report.json presign (we handle above)
                        continue

                    # ------------------
                    # VIDEO OUTPUTS (overlay/dots/skeleton)
                    # ------------------
                    st.markdown(f"#### 🎬 {name}")

                    if not s3_key_exists(out_key):
                        st.warning(f"Output key not found yet: {out_key}")
                        continue

                    # filename guess
                    out_fname = out_key.split("/")[-1] or f"{name}.mp4"
                    out_ct = guess_content_type(out_fname)

                    slot = f"out_{name_lc}"
                    if st.button(f"⬇️ Prepare download link: {name}", key=f"btn_prepare_{slot}_{jid}"):
                        url = presigned_get_url(
                            out_key,
                            expires=3600,
                            filename=out_fname,
                            content_type=out_ct,
                        )
                        remember_url(slot, jid, url)

                    url_saved = get_remembered_url(slot, jid)
                    if url_saved:
                        st.link_button(f"Download {name} — file", url_saved)
