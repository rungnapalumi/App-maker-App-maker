import os
import io
import json
import uuid
from datetime import datetime, timezone

import streamlit as st
import boto3
from botocore.exceptions import ClientError

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Submit Job (Legacy)", layout="wide")
st.title("🚀 Submit Job (Legacy) — dots / skeleton / overlay (NO REPORT)")

st.caption(
    "โหมดนี้ใช้ legacy queue แบบเดิม: เขียน jobs/pending/<job_id>.json ให้ worker ตัวเดิมหยิบงานได้ทันที"
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

# session state
if "last_job_id" not in st.session_state:
    st.session_state["last_job_id"] = ""
if "download_urls" not in st.session_state:
    st.session_state["download_urls"] = {}  # job_id -> {label:url}


# =========================
# Helpers
# =========================
def new_job_id() -> str:
    """
    ✅ match old worker pattern: YYYYMMDD_HHMMSS__xxxxx
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:5]
    return f"{ts}__{rand}"


def safe_filename(name: str) -> str:
    n = (name or "").strip().replace("\\", "/").split("/")[-1]
    keep = []
    for ch in n:
        if ch.isalnum() or ch in ("-", "_", ".", " "):
            keep.append(ch)
    n2 = "".join(keep).strip().replace(" ", "_")
    return n2 or "input.mp4"


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
    if fn.endswith(".json"):
        return "application/json"
    return "application/octet-stream"


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


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=AWS_BUCKET, Key=key)
        return True
    except Exception:
        return False


def presigned_get_url(
    key: str,
    expires: int = 3600,
    filename: str | None = None,
    content_type: str | None = None,
) -> str:
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


def list_output_files(job_id: str) -> list[str]:
    """
    ✅ สแกน output แบบเดิม: jobs/<job_id>/output/
    ใช้แทน status.json ในกรณี worker ไม่อัปเดต status
    """
    prefix = f"jobs/{job_id}/output/"
    keys: list[str] = []
    try:
        resp = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=prefix)
        for it in resp.get("Contents", []) or []:
            k = it.get("Key")
            if isinstance(k, str) and k.endswith("/") is False:
                keys.append(k)
    except Exception:
        pass
    return keys


def build_downloads_from_keys(job_id: str, keys: list[str]) -> dict[str, str]:
    """
    สร้าง label->url จากไฟล์ที่เจอใน output
    """
    out: dict[str, str] = {}
    for k in keys:
        fname = k.split("/")[-1] or "output"
        ctype = guess_content_type(fname)
        url = presigned_get_url(k, expires=3600, filename=fname, content_type=ctype)
        label = f"⬇️ Download {fname}"
        out[label] = url
    return out


# =========================
# UI: Submit
# =========================
st.subheader("1) Upload video + submit (legacy queue)")

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "webm"])
    note = st.text_input("Note (optional)", value="")

with col2:
    st.markdown("### Modes (legacy)")
    mode_overlay = st.checkbox("overlay", value=True)
    mode_dots = st.checkbox("dots", value=False)
    mode_skeleton = st.checkbox("skeleton", value=False)

modes: list[str] = []
if mode_overlay:
    modes.append("overlay")
if mode_dots:
    modes.append("dots")
if mode_skeleton:
    modes.append("skeleton")

if len(modes) == 0:
    st.warning("เลือกอย่างน้อย 1 mode (overlay / dots / skeleton)")

st.caption("Submit จะเขียน jobs/pending/<job_id>.json (legacy) ให้ worker ตัวเดิมหยิบงาน")

if st.button("🚀 Submit job", disabled=(uploaded is None or len(modes) == 0)):
    try:
        job_id = new_job_id()

        filename = safe_filename(uploaded.name if uploaded else "input.mp4")
        content_type = guess_content_type(filename)

        # 1) upload input
        input_key = f"jobs/{job_id}/input/{filename}"
        s3_put_bytes(input_key, uploaded.getvalue(), content_type=content_type)

        # 2) write status.json (หน้า UI จะอ่าน)
        s3_put_json(f"jobs/{job_id}/status.json", {"status": "queued", "job_id": job_id})

        # 3) ✅ legacy pending file (สิ่งที่ worker เก่าหยิบ)
        # ใส่ field ให้กว้าง ๆ เผื่อ worker รุ่นต่างกัน (unknown fields worker ignore)
        pending_payload = {
            "job_id": job_id,
            "input_key": input_key,
            "modes": modes,
            "note": note,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # compatibility helpers (บางเวอร์ชันชอบมี job_key)
            "job_key": f"jobs/{job_id}/job.json",
        }
        s3_put_json(f"jobs/pending/{job_id}.json", pending_payload)

        # 4) optional job.json (บางเวอร์ชันใช้, บางเวอร์ชันไม่ใช้)
        s3_put_json(f"jobs/{job_id}/job.json", pending_payload)

        st.session_state["last_job_id"] = job_id
        st.success("Submitted ✅ (legacy pending written)")
        st.code(json.dumps(pending_payload, ensure_ascii=False, indent=2))

        st.info(f"Legacy queue file: jobs/pending/{job_id}.json")

    except ClientError as e:
        st.error("Submit failed (S3 ClientError)")
        st.exception(e)
    except Exception as e:
        st.error("Submit failed")
        st.exception(e)


st.divider()
st.subheader("2) Verify job + downloads (read-only)")

job_id_check = st.text_input("Job ID to check", value=st.session_state.get("last_job_id", ""))

# show saved links (ไม่หายตอน rerun)
jid = (job_id_check or "").strip()
if jid and jid in st.session_state["download_urls"] and st.session_state["download_urls"][jid]:
    st.subheader("Downloads (saved links)")
    for label, url in st.session_state["download_urls"][jid].items():
        st.link_button(label, url)

if st.button("Check now"):
    if not jid:
        st.warning("Please enter job_id")
    else:
        # 1) try show status.json if exists
        status_key = f"jobs/{jid}/status.json"
        if s3_key_exists(status_key):
            try:
                obj = s3.get_object(Bucket=AWS_BUCKET, Key=status_key)
                data = obj["Body"].read().decode("utf-8", errors="replace")
                status_obj = json.loads(data)
                st.markdown("### status.json")
                st.json(status_obj)
            except Exception as e:
                st.warning("อ่าน status.json ไม่สำเร็จ แต่จะสแกน output ให้แทน")
        else:
            st.info("ไม่พบ status.json (ไม่เป็นไร) — จะสแกน output ให้แทน")

        # 2) ✅ scan output folder (ของจริง)
        out_keys = list_output_files(jid)
        if not out_keys:
            st.info("ยังไม่พบไฟล์ใน jobs/<job_id>/output/ (ยังไม่ออก หรือ worker ยังไม่หยิบงาน)")
        else:
            st.success(f"พบ output {len(out_keys)} file(s) ✅")
            urls = build_downloads_from_keys(jid, out_keys)
            st.session_state["download_urls"][jid] = urls  # save links

            st.subheader("Downloads")
            for label, url in urls.items():
                st.link_button(label, url)
