# 2_Submit_Job.py
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

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


# =========================
# Helpers
# =========================
def new_job_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"{ts}_{rand}"


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


def build_job_manifest(job_id: str, input_key: str, modes: List[str], note: str = "") -> dict:
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
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """
    ✅ Force download (not open in browser tab) by setting:
      ResponseContentDisposition = attachment; filename="..."
    """
    params: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Key": key}

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


def read_status_json(job_id: str) -> Optional[dict]:
    key = f"jobs/{job_id}/status.json"
    try:
        obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
        data = obj["Body"].read().decode("utf-8", errors="replace")
        return json.loads(data)
    except Exception:
        return None


# =========================
# ✅ Presentation Analysis helpers (REPORT ONLY)
# =========================
def normalize_base_url(url: str) -> str:
    u = (url or "").strip()
    return u[:-1] if u.endswith("/") else u


def build_pa_ui_url(pa_base_url: str, job_id: str, lang: str) -> str:
    base = normalize_base_url(pa_base_url)
    return f"{base}/?job_id={job_id}&lang={lang}"


def try_generate_report_via_pa_api(pa_base_url: str, job_id: str, lang: str) -> Optional[dict]:
    """
    พยายามเรียก API ของ Presentation Analysis เพื่อ generate report แล้วคืน key ใน S3
    - ถ้าโปรเจกต์คุณครูมี API อยู่แล้ว -> ให้ปรับ endpoint ให้ตรง
    - ถ้าไม่มี API -> คืน None (ไม่ทำให้หน้าอื่นพัง)
    """
    if requests is None:
        return None

    base = normalize_base_url(pa_base_url)
    endpoint = f"{base}/api/generate_report"  # ✅ ปรับได้ตามของจริง

    try:
        r = requests.get(endpoint, params={"job_id": job_id, "lang": lang}, timeout=60)
        if r.status_code != 200:
            return None
        ct = (r.headers.get("content-type", "") or "").lower()
        if "application/json" not in ct:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def extract_report_s3_key(api_response: dict) -> Optional[str]:
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

modes: List[str] = []
if mode_overlay:
    modes.append("overlay")
if mode_dots:
    modes.append("dots")
if mode_skeleton:
    modes.append("skeleton")
if mode_report:
    modes.append("report")

st.caption("modes จะถูกเขียนลง jobs/<job_id>/job.json เพื่อให้ worker อ่านไปทำงาน")


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

        # 3) initial status
        s3_put_json(f"jobs/{job_id}/status.json", {"status": "queued", "job_id": job_id})

        # ✅ จำ job ล่าสุดไว้ auto-fill ด้านล่าง
        st.session_state["last_job_id"] = job_id

        # ✅ เคลียร์ URL เก่าของ job ก่อนหน้า (กันกดแล้วไปโหลดผิด)
        for k in list(st.session_state.keys()):
            if k.startswith("dl_url__") or k.startswith("report_"):
                # ลบเฉพาะถ้าเป็นของหน้า download (ปลอดภัย)
                pass

        st.success("Submitted ✅")
        st.code(json.dumps(job, ensure_ascii=False, indent=2))

        st.markdown("### Next")
        st.write(f"✅ ตอนนี้มี job ใน S3 แล้ว: `jobs/{job_id}/...`")

        st.info(
            "ถ้า worker ทำงานอยู่และอ่าน jobs/<job_id>/job.json ได้ "
            "มันจะเขียน output ลง jobs/<job_id>/output/... แล้วคุณไปกดดาวน์โหลดในข้อ 2/3 ด้านล่าง"
        )

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

# ✅ เก็บ status_obj ใน session_state เพื่อไม่ต้องกดแล้วหาย
if "last_status_obj" not in st.session_state:
    st.session_state["last_status_obj"] = None
if "last_checked_job_id" not in st.session_state:
    st.session_state["last_checked_job_id"] = ""

if st.button("Check status.json"):
    if not job_id_check.strip():
        st.warning("Please enter job_id")
    else:
        jid = job_id_check.strip()
        status_obj = read_status_json(jid)
        if status_obj is None:
            st.error("Cannot read status.json (not found or parse error)")
        else:
            st.session_state["last_status_obj"] = status_obj
            st.session_state["last_checked_job_id"] = jid

# ✅ แสดงผลจาก state (ไม่หายหลัง rerun)
jid = (st.session_state.get("last_checked_job_id") or "").strip()
status_obj = st.session_state.get("last_status_obj")

if jid and isinstance(status_obj, dict):
    st.json(status_obj)

    outputs = (status_obj or {}).get("outputs") or {}
    if isinstance(outputs, dict) and len(outputs) > 0:
        st.subheader("3) Downloads")

        # --- loop outputs ---
        for name, out_key in outputs.items():
            if not isinstance(out_key, str) or not out_key.strip():
                continue

            out_key = out_key.strip()
            name_lc = str(name).lower().strip()

            # =========================================================
            # ✅ REPORT (TH/EN) — FIX: เก็บ URL ใน session_state กันปุ่มหาย
            # =========================================================
            if name_lc == "report":
                st.markdown("#### 📄 Report (from Presentation Analysis)")

                pa = normalize_base_url(PA_BASE_URL)
                th_url_key = f"report_th_url__{jid}"
                en_url_key = f"report_en_url__{jid}"
                th_fallback_key = f"report_th_fallback__{jid}"
                en_fallback_key = f"report_en_fallback__{jid}"

                col_th, col_en = st.columns(2)

                with col_th:
                    if st.button("⬇️ Generate/Fetch report (TH)", key=f"gen_report_th_{jid}"):
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
                            st.session_state[th_url_key] = url
                            st.session_state[th_fallback_key] = None
                        else:
                            st.session_state[th_url_key] = None
                            st.session_state[th_fallback_key] = build_pa_ui_url(pa, jid, "th")

                    th_url = st.session_state.get(th_url_key)
                    if th_url:
                        st.link_button("Download report (TH) — file", th_url)
                    else:
                        fb = st.session_state.get(th_fallback_key)
                        if fb:
                            st.link_button("Open Presentation Analysis (TH)", fb)

                with col_en:
                    if st.button("⬇️ Generate/Fetch report (EN)", key=f"gen_report_en_{jid}"):
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
                            st.session_state[en_url_key] = url
                            st.session_state[en_fallback_key] = None
                        else:
                            st.session_state[en_url_key] = None
                            st.session_state[en_fallback_key] = build_pa_ui_url(pa, jid, "en")

                    en_url = st.session_state.get(en_url_key)
                    if en_url:
                        st.link_button("Download report (EN) — file", en_url)
                    else:
                        fb = st.session_state.get(en_fallback_key)
                        if fb:
                            st.link_button("Open Presentation Analysis (EN)", fb)

                # ข้าม item นี้ (ไม่ให้ใช้ presigned ของ report.json เดิม)
                continue

            # =========================================================
            # ✅ overlay/dots/skeleton — FIX: เก็บ URL ไว้ ไม่หายหลัง rerun
            # =========================================================
            if not s3_key_exists(out_key):
                st.warning(f"Output key not found yet: {name} -> {out_key}")
                continue

            # เก็บ url ต่อ output key (unique)
            safe_name = "".join([c if c.isalnum() or c in "_-" else "_" for c in str(name_lc)])
            url_state_key = f"dl_url__{jid}__{safe_name}"

            # สร้าง URL แล้วเก็บไว้ (รอบแรก)
            if url_state_key not in st.session_state:
                # เดา filename ตาม key จริง จะถูกกว่า
                fname = out_key.split("/")[-1] or f"{safe_name}.mp4"
                ct = guess_content_type(fname)
                st.session_state[url_state_key] = presigned_get_url(
                    out_key,
                    expires=3600,
                    filename=fname,
                    content_type=ct,
                )

            url = st.session_state.get(url_state_key)
            label = f"⬇️ Download {name}"

            if hasattr(st, "link_button"):
                st.link_button(label, url)
            else:
                st.markdown(f"[{label}]({url})")

    else:
        st.info("ยังไม่มี outputs ใน status.json (รอ worker เขียน outputs ก่อน)")
else:
    st.caption("ใส่ job_id แล้วกด Check status.json เพื่อดู outputs และดาวน์โหลด")
