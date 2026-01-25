import os
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3
from botocore.exceptions import ClientError

# OPTIONAL: ใช้เรียก Presentation Analysis API (ถ้าไม่มี requests ก็ไม่พัง)
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
    "หน้านี้เป็นหน้าใหม่แยกจาก app.py เดิม: ทำแค่อัปโหลด + สร้าง job.json/status.json ใน S3 (ไม่ยุ่งโค้ดเดิม)"
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

# session keys
if "last_job_id" not in st.session_state:
    st.session_state["last_job_id"] = ""
if "download_urls" not in st.session_state:
    st.session_state["download_urls"] = {}  # dict[str, dict[str, str]]  job_id -> {label:url}
if "report_urls" not in st.session_state:
    st.session_state["report_urls"] = {}     # dict[str, dict[str, str]]  job_id -> {TH:url, EN:url}


# =========================
# Helpers
# =========================
def new_job_id() -> str:
    """
    ✅ IMPORTANT: match the OLD worker expectation:
       YYYYMMDD_HHMMSS__xxxxx  (double underscore)
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def safe_filename(name: str) -> str:
    n = (name or "").strip().replace("\\", "/").split("/")[-1]
    # กันชื่อแปลก ๆ
    keep = []
    for ch in n:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
    n2 = "".join(keep).strip().replace(" ", "_")
    return n2 or "input.mp4"


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


def build_job_manifest(job_id: str, input_key: str, modes: list[str], note: str = "") -> dict:
    return {
        "job_id": job_id,
        "input_key": input_key,
        "modes": modes,
        "note": note,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": "submit-v2-legacy-compatible",
    }


def presigned_get_url(
    key: str,
    expires: int = 3600,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
    """
    ✅ Force download (not open in browser tab) by setting:
      ResponseContentDisposition = attachment; filename="..."
    """
    params = {"Bucket": AWS_BUCKET, "Key": key}

    if filename:
        params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
    if content_type:
        params["ResponseContentType"] = content_type

    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params=params,
        ExpiresIn=expires,
    )


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


# =========================
# ✅ NEW (REPORT ONLY): Presentation Analysis integration helpers
# =========================
def normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    return u[:-1] if u.endswith("/") else u


def build_pa_ui_url(pa_base_url: str, job_id: str, lang: str) -> str:
    base = normalize_base_url(pa_base_url)
    return f"{base}/?job_id={job_id}&lang={lang}"


def try_generate_report_via_pa_api(pa_base_url: str, job_id: str, lang: str) -> dict | None:
    if requests is None:
        return None

    base = normalize_base_url(pa_base_url)
    endpoint = f"{base}/api/generate_report"

    try:
        r = requests.get(endpoint, params={"job_id": job_id, "lang": lang}, timeout=60)
        if r.status_code != 200:
            return None
        if not r.headers.get("content-type", "").startswith("application/json"):
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
# UI: Submit
# =========================
st.subheader("1) Upload video + create job.json")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "webm"])
    note = st.text_input("Note (optional)", value="")

with col2:
    st.markdown("### Modes to request")
    mode_overlay = st.checkbox("overlay", value=True)
    mode_dots = st.checkbox("dots", value=False)
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

st.caption("modes จะถูกเขียนลง jobs/<job_id>/job.json เพื่อให้ worker อ่านไปทำงาน")

# ✅ สำคัญ: ให้ worker เดิมที่ scan jobs/pending ทำงานได้
write_legacy_pending = st.checkbox(
    "✅ Also write legacy pending file (jobs/pending/<job_id>.json)",
    value=True
)

if st.button("🚀 Submit job", disabled=(uploaded is None)):
    try:
        job_id = new_job_id()

        filename = safe_filename(uploaded.name if uploaded else "input.mp4")
        content_type = guess_content_type(filename)

        # 1) upload input video to S3
        input_key = f"jobs/{job_id}/input/{filename}"
        video_bytes = uploaded.getvalue()
        s3_put_bytes(input_key, video_bytes, content_type=content_type)

        # 2) write job manifest
        job = build_job_manifest(job_id, input_key, modes=modes, note=note)
        s3_put_json(f"jobs/{job_id}/job.json", job)

        # 3) initial status
        s3_put_json(f"jobs/{job_id}/status.json", {"status": "queued", "job_id": job_id})

        # 4) legacy pending (for OLD worker)
        if write_legacy_pending:
            legacy_pending_key = f"jobs/pending/{job_id}.json"
            legacy_payload = {
                "job_id": job_id,
                "job_key": f"jobs/{job_id}/job.json",
                "input_key": input_key,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            s3_put_json(legacy_pending_key, legacy_payload)

        st.session_state["last_job_id"] = job_id

        st.success("Submitted ✅")
        st.code(json.dumps(job, ensure_ascii=False, indent=2))
        if write_legacy_pending:
            st.info(f"Wrote legacy pending: jobs/pending/{job_id}.json")

        st.markdown("### Next")
        st.write(f"✅ ตอนนี้มี job ใน S3 แล้ว: `jobs/{job_id}/...`")

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)


st.divider()
st.subheader("2) Verify job exists (read-only)")

job_id_check = st.text_input("Job ID to check", value=st.session_state.get("last_job_id", ""))

pa_default = os.getenv("PRESENTATION_ANALYSIS_URL", "https://presentation-analysis.onrender.com")
PA_BASE_URL = st.text_input("Presentation Analysis URL (for report)", value=pa_default)

# ✅ แสดง download links ที่เคยสร้างไว้ (ไม่ให้หายตอน rerun)
jid_tmp = (job_id_check or "").strip()
if jid_tmp:
    if jid_tmp in st.session_state["download_urls"] and st.session_state["download_urls"][jid_tmp]:
        st.subheader("3) Downloads (saved links)")
        for label, url in st.session_state["download_urls"][jid_tmp].items():
            st.link_button(label, url)

    if jid_tmp in st.session_state["report_urls"] and st.session_state["report_urls"][jid_tmp]:
        st.subheader("📄 Report links (saved)")
        ru = st.session_state["report_urls"][jid_tmp]
        if "TH" in ru:
            st.link_button("⬇️ Download report (TH)", ru["TH"])
        if "EN" in ru:
            st.link_button("⬇️ Download report (EN)", ru["EN"])


if st.button("Check status.json"):
    if not job_id_check.strip():
        st.warning("Please enter job_id")
    else:
        jid = job_id_check.strip()
        key = f"jobs/{jid}/status.json"
        try:
            obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
            data = obj["Body"].read().decode("utf-8", errors="replace")
            status_obj = json.loads(data)

            st.json(status_obj)

            outputs = (status_obj or {}).get("outputs") or {}
            if not (isinstance(outputs, dict) and len(outputs) > 0):
                st.info("ยังไม่มี outputs ใน status.json (รอ worker เขียน outputs ก่อน)")
            else:
                st.subheader("3) Downloads")

                # เตรียม dict เก็บ url เพื่อไม่ให้หาย
                if jid not in st.session_state["download_urls"]:
                    st.session_state["download_urls"][jid] = {}

                for name, out_key in outputs.items():
                    if not isinstance(out_key, str) or not out_key.strip():
                        continue
                    out_key = out_key.strip()

                    if not s3_key_exists(out_key):
                        st.warning(f"Output key not found yet: {name} -> {out_key}")
                        continue

                    # ตั้งชื่อไฟล์ตาม key จริง (ไม่เดาเป็น mp4 เสมอ)
                    real_name = out_key.split("/")[-1] or str(name)
                    ctype = guess_content_type(real_name)

                    url = presigned_get_url(
                        out_key,
                        expires=3600,
                        filename=real_name,
                        content_type=ctype,
                    )

                    label = f"⬇️ Download {name} ({real_name})"
                    st.session_state["download_urls"][jid][label] = url
                    st.link_button(label, url)

        except ClientError as e:
            st.error("Cannot read status.json")
            st.exception(e)
        except Exception as e:
            st.error("Failed to parse status.json")
            st.exception(e)


st.divider()
st.subheader("4) Report (from Presentation Analysis)")

st.caption("กดแล้วลิงก์จะถูกจำไว้ด้านบน (ไม่หายตอน rerun)")

col_th, col_en = st.columns(2)

with col_th:
    if st.button("⬇️ Generate/Download report (TH)", key="gen_report_th"):
        jid = (job_id_check or "").strip()
        if not jid:
            st.warning("กรอก Job ID ก่อน")
        else:
            pa = normalize_base_url(PA_BASE_URL)
            api_res = try_generate_report_via_pa_api(pa, jid, "th")
            report_key = extract_report_s3_key(api_res) if isinstance(api_res, dict) else None

            if report_key and s3_key_exists(report_key):
                fname = report_key.split("/")[-1] or "report_th.pdf"
                url = presigned_get_url(report_key, 3600, fname, guess_content_type(fname))
                st.session_state["report_urls"].setdefault(jid, {})
                st.session_state["report_urls"][jid]["TH"] = url
                st.success("Report ready ✅ (link saved above)")
            else:
                st.info("ยังไม่พบ report key จาก API — เปิดหน้า Presentation Analysis")
                st.link_button("Open Presentation Analysis (TH)", build_pa_ui_url(pa, jid, "th"))

with col_en:
    if st.button("⬇️ Generate/Download report (EN)", key="gen_report_en"):
        jid = (job_id_check or "").strip()
        if not jid:
            st.warning("กรอก Job ID ก่อน")
        else:
            pa = normalize_base_url(PA_BASE_URL)
            api_res = try_generate_report_via_pa_api(pa, jid, "en")
            report_key = extract_report_s3_key(api_res) if isinstance(api_res, dict) else None

            if report_key and s3_key_exists(report_key):
                fname = report_key.split("/")[-1] or "report_en.pdf"
                url = presigned_get_url(report_key, 3600, fname, guess_content_type(fname))
                st.session_state["report_urls"].setdefault(jid, {})
                st.session_state["report_urls"][jid]["EN"] = url
                st.success("Report ready ✅ (link saved above)")
            else:
                st.info("ยังไม่พบ report key จาก API — เปิดหน้า Presentation Analysis")
                st.link_button("Open Presentation Analysis (EN)", build_pa_ui_url(pa, jid, "en"))
