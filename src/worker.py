# worker.py — AI People Reader Worker
#
# หน้าที่:
#   - poll หา "jobs/pending/*.json" ใน S3
#   - ดึง json job ขึ้นมา แล้ว mark ว่า processing
#   - ทำงาน (ตัวอย่างนี้: copy วิดีโอจาก video_key -> result_video_key)
#   - ถ้าสำเร็จ: ย้าย job ไป "jobs/finished/"
#   - ถ้าล้มเหลว: ย้าย job ไป "jobs/failed/" พร้อมเขียน error ลงใน json
#
# NOTE:
#   - โครงสร้าง job JSON ที่ EXPECT:
#       {
#           "job_id": "20260113_132520__abcd1",
#           "status": "pending",
#           "video_key": "uploads/xxx.mp4",
#           "result_video_key": "results/xxx_processed.mp4",
#           "created_at": "...",
#           "updated_at": "..."
#       }
#
#   - ถ้า app.py ใช้ชื่อ field ต่างจากนี้
#     ให้แก้ชื่อ field ใน process_job() ให้ตรง
#

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

# ----------------------------------------------------------
# Config / S3 client
# ----------------------------------------------------------

AWS_BUCKET = os.environ.get("AWS_BUCKET") or os.environ.get("S3_BUCKET")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
JOB_POLL_INTERVAL = float(os.environ.get("JOB_POLL_INTERVAL", "5"))  # วินาที

if not AWS_BUCKET:
    raise RuntimeError("Missing AWS_BUCKET / S3_BUCKET environment variable")

s3 = boto3.client("s3", region_name=AWS_REGION)

print("====== AI People Reader Worker ======", flush=True)
print(f"Using bucket: {AWS_BUCKET}", flush=True)
print(f"Region     : {AWS_REGION}", flush=True)
print(f"Poll every : {JOB_POLL_INTERVAL} seconds", flush=True)


# ----------------------------------------------------------
# Helper: timestamp
# ----------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------
# Helper: S3 JSON I/O
# ----------------------------------------------------------

def s3_get_json(key: str) -> Dict[str, Any]:
    """
    โหลด JSON จาก S3 แล้ว parse เป็น dict
    """
    print(f"[s3_get_json] key={key}", flush=True)
    obj = s3.get_object(Bucket=AWS_BUCKET, Key=key)
    body = obj["Body"].read()
    return json.loads(body.decode("utf-8"))


def s3_put_json(key: str, data: Dict[str, Any]) -> None:
    """
    เซฟ dict เป็น JSON ลง S3
    """
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    print(f"[s3_put_json] key={key} size={len(body)} bytes", flush=True)
    s3.put_object(Bucket=AWS_BUCKET, Key=key, Body=body, ContentType="application/json")


# ----------------------------------------------------------
# Helper: copy / move objects ใน S3
# ----------------------------------------------------------

def copy_object(src_key: str, dst_key: str) -> None:
    """
    copy object ใน bucket เดียวกัน
    ใช้รูปแบบ CopySource แบบ dict ให้ถูกต้องตาม S3 spec
    """
    print(f"[copy_object] {src_key} -> {dst_key}", flush=True)
    s3.copy_object(
        Bucket=AWS_BUCKET,
        Key=dst_key,
        CopySource={
            "Bucket": AWS_BUCKET,
            "Key": src_key,
        },
    )


def move_object(src_key: str, dst_key: str) -> None:
    """
    move object ใน bucket เดียวกัน (copy + delete)
    """
    print(f"[move_object] {src_key} -> {dst_key}", flush=True)
    # copy ก่อน
    copy_object(src_key, dst_key)
    # ลบต้นทาง
    print(f"[move_object] delete source {src_key}", flush=True)
    s3.delete_object(Bucket=AWS_BUCKET, Key=src_key)


def copy_video(video_key: str, result_video_key: str) -> None:
    """
    ตัวอย่าง simple work: แค่ copy วิดีโอจาก video_key -> result_video_key
    (ตอนหลัง Rung อยากเอาไปใส่ขั้นตอนประมวลผลจริง เช่น clear, overlay ฯลฯ
     ก็เอา logic มาแทนส่วนนี้ได้เลย)
    """
    print(f"[copy_video] {video_key} -> {result_video_key}", flush=True)
    copy_object(video_key, result_video_key)


# ----------------------------------------------------------
# Helper: หา pending job
# ----------------------------------------------------------

PENDING_PREFIX = "jobs/pending/"
PROCESSING_PREFIX = "jobs/processing/"
FINISHED_PREFIX = "jobs/finished/"
FAILED_PREFIX = "jobs/failed/"


def find_one_pending_job_key() -> Optional[str]:
    """
    หา job แรกใน jobs/pending/ (ถ้าไม่มี return None)
    """
    print(f"[find_one_pending_job_key] scanning prefix={PENDING_PREFIX}", flush=True)
    resp = s3.list_objects_v2(
        Bucket=AWS_BUCKET,
        Prefix=PENDING_PREFIX,
        MaxKeys=1,
    )
    contents = resp.get("Contents")
    if not contents:
        print("[find_one_pending_job_key] no pending jobs", flush=True)
        return None

    # เอา key แรก
    key = contents[0]["Key"]
    print(f"[find_one_pending_job_key] found job key={key}", flush=True)
    return key


# ----------------------------------------------------------
# Core: process one job
# ----------------------------------------------------------

def process_job(pending_key: str) -> None:
    """
    ประมวลผล job 1 ตัว
    `pending_key` คือ key ของ json ที่อยู่ใน jobs/pending/
    """

    print(f"[process_job] start pending_key={pending_key}", flush=True)

    # อ่าน job JSON
    job_data = s3_get_json(pending_key)

    job_id = job_data.get("job_id")
    if not job_id:
        # ถ้าไม่มี job_id ให้ derive จากชื่อไฟล์
        job_id = os.path.splitext(os.path.basename(pending_key))[0]
        job_data["job_id"] = job_id

    # ดึง field หลักตาม schema
    # ถ้า app.py ใช้ชื่ออื่น ให้แก้ชื่อตรงนี้
    video_key = job_data.get("video_key")
    result_video_key = job_data.get("result_video_key")

    if not video_key:
        raise RuntimeError("Job JSON missing `video_key`")

    if not result_video_key:
        # default path ถ้าไม่ได้ส่งมา
        result_video_key = f"results/{job_id}.mp4"
        job_data["result_video_key"] = result_video_key

    # ------------------------------------------------------
    # 1) mark เป็น processing (ย้าย json ไป jobs/processing)
    # ------------------------------------------------------
    processing_key = pending_key.replace(PENDING_PREFIX, PROCESSING_PREFIX)

    job_data["status"] = "processing"
    job_data["updated_at"] = utc_now_iso()

    # เขียนไปที่ processing/ แล้วค่อยลบ pending/ (move)
    s3_put_json(processing_key, job_data)
    print(f"[process_job] move JSON pending -> processing", flush=True)
    s3.delete_object(Bucket=AWS_BUCKET, Key=pending_key)

    try:
        # --------------------------------------------------
        # 2) ทำงานหลัก — ตอนนี้คือ copy วิดีโอ
        # --------------------------------------------------
        copy_video(video_key, result_video_key)

        # --------------------------------------------------
        # 3) mark finished
        # --------------------------------------------------
        finished_key = processing_key.replace(PROCESSING_PREFIX, FINISHED_PREFIX)
        job_data["status"] = "finished"
        job_data["finished_at"] = utc_now_iso()
        job_data["updated_at"] = job_data["finished_at"]
        job_data["error"] = None

        s3_put_json(finished_key, job_data)
        print(f"[process_job] move JSON processing -> finished", flush=True)
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)

    except Exception as exc:
        # --------------------------------------------------
        # 4) ถ้า error -> failed
        # --------------------------------------------------
        print(f"[process_job] ERROR: {exc}", flush=True)

        failed_key = processing_key.replace(PROCESSING_PREFIX, FAILED_PREFIX)

        job_data["status"] = "failed"
        job_data["error"] = str(exc)
        job_data["failed_at"] = utc_now_iso()
        job_data["updated_at"] = job_data["failed_at"]

        s3_put_json(failed_key, job_data)
        print(f"[process_job] move JSON processing -> failed", flush=True)

        # ลบ processing json ตัวเก่า
        s3.delete_object(Bucket=AWS_BUCKET, Key=processing_key)

        # raise ซ้ำให้เห็นใน logs
        raise


# ----------------------------------------------------------
# Main loop
# ----------------------------------------------------------

def main() -> None:
    print("[main] worker started", flush=True)
    while True:
        try:
            job_key = find_one_pending_job_key()
            if not job_key:
                time.sleep(JOB_POLL_INTERVAL)
                continue

            process_job(job_key)

        except ClientError as ce:
            # error จาก S3 / AWS
            print(f"[main] AWS ClientError: {ce}", flush=True)
            time.sleep(JOB_POLL_INTERVAL)

        except Exception as e:
            # error อื่น ๆ
            print(f"[main] Unexpected error: {e}", flush=True)
            time.sleep(JOB_POLL_INTERVAL)


if __name__ == "__main__":
    main()
