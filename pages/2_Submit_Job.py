import os
import json
import uuid
from datetime import datetime, timezone
from typing import Any

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
st.caption("หน้าใหม่แยกจาก app.py เดิม: อัปโหลด + สร้าง job/status ใน S3 (ไม่ยุ่งโค้ดเดิม)")

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
    # ✅ กลับไปใช้ format เดิมที่ worker เคยใช้: YYYYMMDD_HHMMSS__xxxxx
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


def guess_content_type(filename: str) -> str:
    fn = filename.lower()
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
        "version": "submit-v1",
    }


def presigned_get_url(
    key: str,
    expires: int = 3600,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
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


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


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
    if requests is None:
        return None

    base = normalize_base_url(pa_base_url)
    endpoint = f"{base}/api/generate_report"

    try:
        r = requests.get(endpoint, params={"job_id": job_id, "lang": lang}, timeout=60)
        if r.status_code != 200:
            return None
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
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

legacy_bridge = st.checkbox("✅ Also write legacy pending file (jobs/pending/<job_id>.json)", value=True)


if st.button("🚀 Submit job", disabled=(uploaded is None)):
    try:
        job_id = new_job_id()

        filename = uploaded.name if uploaded else "input.mp4"
        content_type = guess_content_type(filename)

        # 1) upload input video to S3
        input_key = f"jobs/{job_id}/input/{filename}"
        video_bytes = uploaded.getvalue()
        s3_put_bytes(input_key, video_bytes, content_type=content_type)

        # 2) write job manifest
        job = build_job_manifest(job_id, input_key, modes=modes, note=note)
        s3_put_json(f"jobs/{job_id}/job.json", job)

        # ✅ สำคัญ: worker เดิมคุณ “น่าจะหยิบงานจาก pending”
        # และ log ตอนนี้ค้างเพราะ status เป็น queued -> เปลี่ยนเป็น pending
        s3_put_json(f"jobs/{job_id}/status.json", {"status": "pending", "job_id": job_id})

        # ✅ legacy bridge
        if legacy_bridge:
            # ทำเป็นไฟล์ pending ให้ worker เดิมเห็นแน่ ๆ
            s3_put_json(f"jobs/pending/{job_id}.json", job)

        st.session_state["last_job_id"] = job_id

        st.success("Submitted ✅")
        st.write("Job ID:", job_id)
        st.code(json.dumps(job, ensure_ascii=False, indent=2))

        st.info(f"status.json set to: pending  ✅")
        if legacy_bridge:
            st.info(f"legacy pending written: jobs/pending/{job_id}.json ✅")

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)


# =========================
# Verify + Downloads
# =========================
st.divider()
st.subheader("2) Verify job exists (read-only)")

job_id_check = st.text_input("Job ID to check", value=st.session_state.get("last_job_id", ""))

pa_default = os.getenv("PRESENTATION_ANALYSIS_URL", "https://presentation-analysis.onrender.com")
PA_BASE_URL = st.text_input("Presentation Analysis URL (for report)", value=pa_default)

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
            if not isinstance(outputs, dict) or len(outputs) == 0:
                st.info("ยังไม่มี outputs ใน status.json (รอ worker เขียน outputs ก่อน)")
            else:
                st.subheader("3) Downloads")

                for name, out_key in outputs.items():
                    if not isinstance(out_key, str) or not out_key.strip():
                        continue

                    out_key = out_key.strip()
                    name_lc = str(name).lower().strip()

                    if name_lc == "report":
                        st.markdown("#### 📄 Report (from Presentation Analysis)")
                        col_th, col_en = st.columns(2)

                        with col_th:
                            if st.button("⬇️ Prepare report (TH)", key=f"prep_report_th_{jid}"):
                                pa = normalize_base_url(PA_BASE_URL)
                                api_res = try_generate_report_via_pa_api(pa, jid, "th")
                                report_key = extract_report_s3_key(api_res) if isinstance(api_res, dict) else None
                                if report_key and s3_key_exists(report_key):
                                    fname = report_key.split("/")[-1] or "report_th.pdf"
                                    url = presigned_get_url(report_key, filename=fname, content_type=guess_content_type(fname))
                                    st.session_state["report_th_url"] = url
                                else:
                                    st.session_state["report_th_url"] = build_pa_ui_url(pa, jid, "th")

                            if st.session_state.get("report_th_url"):
                                st.link_button("Download/Open (TH)", st.session_state["report_th_url"])

                        with col_en:
                            if st.button("⬇️ Prepare report (EN)", key=f"prep_report_en_{jid}"):
                                pa = normalize_base_url(PA_BASE_URL)
                                api_res = try_generate_report_via_pa_api(pa, jid, "en")
                                report_key = extract_report_s3_key(api_res) if isinstance(api_res, dict) else None
                                if report_key and s3_key_exists(report_key):
                                    fname = report_key.split("/")[-1] or "report_en.pdf"
                                    url = presigned_get_url(report_key, filename=fname, content_type=guess_content_type(fname))
                                    st.session_state["report_en_url"] = url
                                else:
                                    st.session_state["report_en_url"] = build_pa_ui_url(pa, jid, "en")

                            if st.session_state.get("report_en_url"):
                                st.link_button("Download/Open (EN)", st.session_state["report_en_url"])

                        continue

                    if not s3_key_exists(out_key):
                        st.warning(f"Output key not found yet: {name} -> {out_key}")
                        continue

                    fname = out_key.split("/")[-1] or f"{name}.mp4"
                    url = presigned_get_url(out_key, filename=fname, content_type=guess_content_type(fname))
                    st.link_button(f"⬇️ Download {name}", url)

        except ClientError as e:
            st.error("Cannot read status.json")
            st.exception(e)
        except Exception as e:
            st.error("Failed to parse status.json")
            st.exception(e)
