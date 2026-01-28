# src/report_worker.py
# ------------------------------------------------------------
# AI People Reader - Report Worker (CV Proxy version)
#
# - Polls S3: jobs/pending/*.json
# - Handles only mode=="report"
# - Downloads input video
# - Normalizes video with FFmpeg
# - Analyzes from REAL video using Classical CV (NO ML MODEL):
#     2) Uprightness
#     3) Stance
#     4) Motion Dynamics (proxy)
# - Generates DOCX report (NO PDF)
# - Uploads outputs to S3
# - Moves job json to finished / failed
#
# SAFE FOR RENDER WORKER
# ------------------------------------------------------------

import os
import io
import json
import time
import shutil
import tempfile
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List

import boto3

# optional libs (same as dots/skeleton worker)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# docx
try:
    from docx import Document  # type: ignore
    from docx.shared import Pt, Inches  # type: ignore
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK  # type: ignore
except Exception:
    Document = None  # type: ignore

# ⭐ Classical CV proxy analysis
from cv_proxy_pose import analyze_cv_proxy

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("report_worker")

# ------------------------------------------------------------
# S3 config
# ------------------------------------------------------------
AWS_BUCKET = os.getenv("AWS_BUCKET") or os.getenv("S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

JOBS_PENDING_PREFIX = "jobs/pending/"
JOBS_PROCESSING_PREFIX = "jobs/processing/"
JOBS_FINISHED_PREFIX = "jobs/finished/"
JOBS_FAILED_PREFIX = "jobs/failed/"

JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "10"))

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET / S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)
# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _t(lang: str, en: str, th: str) -> str:
    lang = (lang or "en").strip().lower()
    return th if lang.startswith("th") else en


def format_seconds_to_mmss(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds or 0.0))
    mm = int(total_seconds // 60)
    ss = int(round(total_seconds - mm * 60))
    if ss == 60:
        mm += 1
        ss = 0
    return f"{mm:02d}:{ss:02d}"


def get_video_duration_seconds(video_path: str) -> float:
    if cv2 is None:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.0
    return float(frames / fps)


# ------------------------------------------------------------
# FFmpeg normalize (always)
# ------------------------------------------------------------

def ensure_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("FFmpeg not found in PATH. Add ffmpeg to worker environment.")
    return ff


def normalize_video_with_ffmpeg(input_path: str, output_path: str, fps: int = 30) -> None:
    ff = ensure_ffmpeg()
    cmd = [
        ff, "-y",
        "-i", input_path,
        "-vf", f"fps={fps},format=yuv420p",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        "-an",
        output_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Soft-fail: if ffmpeg returns non-zero but output exists and non-empty -> continue
    if p.returncode != 0:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            tail = (p.stderr or "")[-1200:]
            log.warning("FFmpeg normalize returned non-zero but output exists; continuing. stderr tail:\n%s", tail)
            return
        tail = (p.stderr or "")[-1800:]
        raise RuntimeError(f"FFmpeg normalize failed. stderr tail:\n{tail}")


# ------------------------------------------------------------
# S3 helpers
# ------------------------------------------------------------

def s3_get_json(key: str) -> dict:
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))


def s3_put_json(key: str, data: dict):
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        ContentType="application/json; charset=utf-8",
    )


def s3_download_to_file(key: str, local_path: str):
    s3.download_file(AWS_BUCKET, key, local_path)


def s3_put_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_copy_delete(src_key: str, dst_key: str):
    s3.copy_object(Bucket=AWS_BUCKET, CopySource={"Bucket": AWS_BUCKET, "Key": src_key}, Key=dst_key)
    s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)


def list_job_json_keys(prefix: str, limit: int = 200) -> list:
    keys: List[str] = []
    token = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": AWS_BUCKET, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for it in resp.get("Contents", []):
            k = it.get("Key", "")
            if k.endswith(".json"):
                keys.append(k)
                if len(keys) >= limit:
                    return keys
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


# ------------------------------------------------------------
# Job schema (tolerant)
# ------------------------------------------------------------

def is_report_job(job: dict) -> bool:
    jt = (job.get("type") or job.get("job_type") or job.get("mode") or "").strip().lower()
    return jt in ("report", "presentation_report", "presentation-analysis-report")


def job_status(job: dict) -> str:
    return (job.get("status") or "").strip().lower()


def set_status(job: dict, status: str, message: str = "") -> dict:
    job["status"] = status
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    if message:
        job["message"] = message
    return job


def job_lang(job: dict) -> str:
    lang = (job.get("lang") or job.get("language") or "").strip().lower()
    return "th" if lang.startswith("th") else "en"


def resolve_video_key(job: dict, job_id: str) -> str:
    for f in ("video_s3_key", "input_video_key", "input_key", "s3_key"):
        if job.get(f):
            return str(job[f])
    # fallback
    return f"jobs/pending/{job_id}/input/input.mp4"
# ------------------------------------------------------------
# Core processing
# ------------------------------------------------------------

def process_report_job(job_key: str):
    job = s3_get_json(job_key)
    job_id = str(job.get("job_id") or job.get("id") or os.path.splitext(os.path.basename(job_key))[0])

    if not is_report_job(job):
        return

    stt = job_status(job)
    if stt in ("processing", "finished", "done", "failed", "error"):
        return

    lang = job_lang(job)

    log.info("Processing report job %s (%s)", job_id, lang)
    job = set_status(job, "processing", "Generating report (CV proxy)…")
    s3_put_json(job_key, job)

    tmp_dir = tempfile.mkdtemp(prefix=f"report_{job_id}_")

    try:
        # ----------------------------------------------------
        # 1) Download video
        # ----------------------------------------------------
        video_key = resolve_video_key(job, job_id)
        ext = os.path.splitext(video_key)[1] or ".mp4"
        raw_video_path = os.path.join(tmp_dir, "input_raw" + ext)

        log.info("Downloading video s3://%s/%s", AWS_BUCKET, video_key)
        s3_download_to_file(video_key, raw_video_path)

        # ----------------------------------------------------
        # 2) Normalize video (FFmpeg)
        # ----------------------------------------------------
        normalized_path = os.path.join(tmp_dir, "input_normalized.mp4")
        log.info("Normalizing video with FFmpeg…")
        normalize_video_with_ffmpeg(raw_video_path, normalized_path, fps=30)

        # ----------------------------------------------------
        # 3) Analyze with Classical CV (REAL video)
        # ----------------------------------------------------
        log.info("Running CV proxy analysis (uprightness / stance / motion)…")
        cv_result = analyze_cv_proxy(
            normalized_path,
            lang=lang,
            sample_fps=6.0,
            max_frames=1200,
        )

        # Extract scores
        upright = cv_result["uprightness"]
        stance = cv_result["stance"]
        motion = cv_result["motion_dynamics"]

        # ----------------------------------------------------
        # 4) Prepare report data (2 / 3 / 4)
        # ----------------------------------------------------
        duration_sec = get_video_duration_seconds(normalized_path)
        duration_str = format_seconds_to_mmss(duration_sec)
        analysis_date = datetime.now().strftime("%d-%m-%Y")

        sections = [
            {
                "no": "2.",
                "title_en": "Uprightness (Posture & Upper-Body Alignment)",
                "title_th": "ความตั้งตรงของท่าทาง (ลำตัวส่วนบน)",
                "score": upright["score_0_100"],
                "level": upright["level"],
                "details": upright["details"],
                "note_en": (
                    "Posture is evaluated from body silhouette orientation "
                    "using classical computer vision (no ML model)."
                ),
                "note_th": (
                    "การประเมินท่าทางใช้แนวลำตัวจากเงาร่างกาย "
                    "ด้วยเทคนิคคอมพิวเตอร์วิทัศน์ (ไม่ใช้โมเดล AI)"
                ),
            },
            {
                "no": "3.",
                "title_en": "Stance (Lower-Body Stability & Grounding)",
                "title_th": "ความมั่นคงของการยืน (ช่วงล่าง)",
                "score": stance["score_0_100"],
                "level": stance["level"],
                "details": stance["details"],
                "note_en": (
                    "Stance is estimated from base width and body sway "
                    "derived from silhouette motion."
                ),
                "note_th": (
                    "การประเมินการยืนวิเคราะห์จากความกว้างฐานและการส่ายของลำตัว "
                    "ซึ่งคำนวณจากเงาร่างกาย"
                ),
            },
            {
                "no": "4.",
                "title_en": "Movement Dynamics (Activity & Expressiveness)",
                "title_th": "พลวัตการเคลื่อนไหว (ระดับความเคลื่อนไหว)",
                "score": motion["score_0_100"],
                "level": motion["level"],
                "details": motion["details"],
                "note_en": (
                    "Movement dynamics reflects overall activity level "
                    "based on frame-to-frame motion intensity."
                ),
                "note_th": (
                    "พลวัตการเคลื่อนไหวสะท้อนระดับการขยับร่างกายโดยรวม "
                    "จากความแตกต่างของภาพในแต่ละเฟรม"
                ),
            },
        ]

        # ----------------------------------------------------
        # 5) Store prepared payload in job (for PART 4)
        # ----------------------------------------------------
        job["_report_payload"] = {
            "job_id": job_id,
            "lang": lang,
            "analysis_date": analysis_date,
            "duration_str": duration_str,
            "sections": sections,
            "cv_meta": cv_result.get("meta", {}),
        }

        job = set_status(job, "processing", "Building DOCX report…")
        s3_put_json(job_key, job)

    except Exception as e:
        log.exception("Report job failed: %s", e)
        job = set_status(job, "failed", str(e))
        s3_put_json(job_key, job)
        raise

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
# ------------------------------------------------------------
# DOCX builder (simple, stable, no external assets)
# ------------------------------------------------------------

def _docx_set_base_font(doc, lang: str):
    if Document is None:
        return
    style = doc.styles["Normal"]
    style.font.name = "TH Sarabun New" if (lang or "").startswith("th") else "Calibri"
    style.font.size = Pt(11)


def _docx_add_title(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(14)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(14)
    return p


def _docx_add_kv(doc, k: str, v: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    rk = p.add_run(k)
    rk.bold = True
    p.add_run(v)
    return p


def _docx_add_heading(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12)
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(8)
    return p


def _docx_add_numbered(doc, no: str, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_after = Pt(4)
    p.add_run(f"{no}\t{text}")
    try:
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.65))
    except Exception:
        pass
    return p


def _docx_add_bullet(doc, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.65)
    p.paragraph_format.first_line_indent = Inches(-0.15)
    p.paragraph_format.space_after = Pt(4)
    p.add_run("•  " + (text or "").strip())
    return p


def _docx_add_note(doc, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.65)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.italic = True
    return p


def build_docx_from_payload(payload: Dict[str, Any]) -> bytes:
    if Document is None:
        raise RuntimeError("python-docx is required to generate DOCX")

    lang = payload.get("lang", "en")
    job_id = payload.get("job_id", "—")
    analysis_date = payload.get("analysis_date", "—")
    duration_str = payload.get("duration_str", "—")

    doc = Document()
    _docx_set_base_font(doc, lang)

    _docx_add_title(doc, _t(lang, "Character Analysis Report", "รายงานวิเคราะห์บุคลิกภาพ"))

    client_name = str(payload.get("client_name") or payload.get("client") or "—")
    _docx_add_kv(doc, _t(lang, "Client Name: ", "ชื่อลูกค้า: "), client_name)
    _docx_add_kv(doc, _t(lang, "Analysis Date: ", "วันที่วิเคราะห์: "), analysis_date)
    _docx_add_kv(doc, _t(lang, "Video Duration: ", "ความยาววิดีโอ: "), duration_str)

    _docx_add_heading(doc, _t(lang, "Detailed Analysis", "Detailed Analysis"))

    # Important disclaimer about Classical CV
    _docx_add_note(
        doc,
        _t(
            lang,
            "Note: This report uses Classical Computer Vision (NO AI model) to estimate posture/stance/motion from silhouette. "
            "Results depend on camera stability, full-body visibility, lighting, and background simplicity.",
            "หมายเหตุ: รายงานนี้ใช้เทคนิค Computer Vision แบบดั้งเดิม (ไม่ใช้โมเดล AI) เพื่อประมาณการท่าทาง/การยืน/การเคลื่อนไหวจากเงาร่างกาย "
            "ผลลัพธ์ขึ้นกับความนิ่งของกล้อง การเห็นร่างกายเต็มตัว แสง และความเรียบง่ายของฉากหลัง",
        ),
    )

    sections: List[Dict[str, Any]] = payload.get("sections", [])

    for sec in sections:
        no = sec.get("no", "")
        title = _t(lang, sec.get("title_en", ""), sec.get("title_th", ""))
        _docx_add_numbered(doc, no, title)

        score = int(sec.get("score") or 0)
        level = str(sec.get("level") or "—")

        _docx_add_bullet(doc, _t(lang, "Scale: ", "ระดับ: ") + f"{level}")
        _docx_add_bullet(doc, _t(lang, "Score: ", "คะแนน: ") + f"{score}/100")

        # Show a few key details (kept short)
        details = sec.get("details") or {}
        if isinstance(details, dict):
            # pick a few common keys
            for k in ("angle_mean_deg", "angle_std_deg", "centroid_x_sway_std", "bottom_width_mean", "motion_energy_mean"):
                if k in details:
                    _docx_add_bullet(doc, f"{k}: {details[k]}")

        note = _t(lang, sec.get("note_en", ""), sec.get("note_th", "")).strip()
        if note:
            _docx_add_note(doc, note)

    # Optional meta/debug summary (kept minimal)
    meta = payload.get("cv_meta") or {}
    if isinstance(meta, dict):
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        _docx_add_heading(doc, _t(lang, "Technical Summary (Debug)", "สรุปเชิงเทคนิค (Debug)"))
        for k in ("sample_fps", "processed_frames", "ok_frames", "duration_sec", "frame_w", "frame_h"):
            v = meta.get(k)
            if v is not None:
                _docx_add_bullet(doc, f"{k}: {v}")

    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# ------------------------------------------------------------
# Main loop runner
# ------------------------------------------------------------

def main_loop():
    log.info("Report worker started. Bucket=%s, pending_prefix=%s", AWS_BUCKET, JOBS_PENDING_PREFIX)

    while True:
        try:
            keys = list_job_json_keys(JOBS_PENDING_PREFIX, limit=200)

            for job_key in sorted(keys):
                try:
                    job = s3_get_json(job_key)
                    if not is_report_job(job):
                        continue

                    stt = job_status(job)
                    if stt in ("processing", "finished", "done", "failed", "error"):
                        continue

                    # Run processing (PART 3 stores _report_payload)
                    process_report_job(job_key)

                    # Re-read (ensure latest)
                    job2 = s3_get_json(job_key)
                    payload = job2.get("_report_payload")
                    if not isinstance(payload, dict):
                        raise RuntimeError("Missing _report_payload (analysis step did not complete)")

                    # Attach client name into payload for DOCX header
                    payload["client_name"] = str(job2.get("client_name") or job2.get("client") or "—")

                    # Build DOCX
                    log.info("Building DOCX for job %s", payload.get("job_id"))
                    docx_bytes = build_docx_from_payload(payload)
                    if not docx_bytes:
                        raise RuntimeError("DOCX generation returned empty bytes")

                    out_docx_key = job2.get("output_docx_key") or f"jobs/output/{payload.get('job_id')}/report_{payload.get('lang')}.docx"
                    out_debug_key = job2.get("output_debug_key") or f"jobs/output/{payload.get('job_id')}/debug_{payload.get('lang')}.json"

                    s3_put_bytes(
                        str(out_docx_key),
                        docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )

                    debug_payload = {
                        "job_id": payload.get("job_id"),
                        "lang": payload.get("lang"),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "outputs": {"docx": str(out_docx_key)},
                        "cv_meta": payload.get("cv_meta", {}),
                    }

                    try:
                        s3_put_bytes(
                            str(out_debug_key),
                            json.dumps(debug_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                            "application/json",
                        )
                    except Exception as e:
                        log.warning("Debug json upload failed: %s", e)

                    # Update job status
                    job2["outputs"] = {"docx": str(out_docx_key), "debug": str(out_debug_key)}
                    job2 = set_status(job2, "finished", "DOCX report generated (CV proxy)")
                    # cleanup internal payload
                    try:
                        del job2["_report_payload"]
                    except Exception:
                        pass
                    s3_put_json(job_key, job2)

                    # Move job json
                    job_id = str(job2.get("job_id") or job2.get("id") or os.path.splitext(os.path.basename(job_key))[0])
                    if job_status(job2) in ("finished", "done"):
                        dst = f"{JOBS_FINISHED_PREFIX}{job_id}.json"
                        s3_copy_delete(job_key, dst)
                        log.info("Moved job to finished: %s", dst)
                    elif job_status(job2) in ("failed", "error"):
                        dst = f"{JOBS_FAILED_PREFIX}{job_id}.json"
                        s3_copy_delete(job_key, dst)
                        log.info("Moved job to failed: %s", dst)

                except Exception as e:
                    log.exception("Job failed for %s: %s", job_key, e)
                    try:
                        jobx = s3_get_json(job_key)
                        jobx = set_status(jobx, "failed", str(e))
                        s3_put_json(job_key, jobx)

                        job_id = str(jobx.get("job_id") or jobx.get("id") or os.path.splitext(os.path.basename(job_key))[0])
                        dst = f"{JOBS_FAILED_PREFIX}{job_id}.json"
                        s3_copy_delete(job_key, dst)
                    except Exception:
                        pass

        except Exception as e:
            log.warning("Polling error: %s", e)

        time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
