import os
import time
import json
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import boto3
import cv2
import numpy as np

# MediaPipe pose
import mediapipe as mp

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    bucket: str
    region: str
    pending_prefix: str = "jobs/pending/"
    jobs_prefix: str = "jobs/"

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        bucket = os.environ.get("S3_BUCKET")
        region = os.environ.get("AWS_REGION", "ap-southeast-1")

        if not bucket:
            raise RuntimeError("S3_BUCKET env var is required")

        return cls(bucket=bucket, region=region)


# -----------------------------------------------------------------------------
# S3 helper
# -----------------------------------------------------------------------------

class S3Client:
    def __init__(self, cfg: WorkerConfig):
        self.cfg = cfg
        self.s3 = boto3.client("s3", region_name=cfg.region)

    # ------ Job discovery -----------------------------------------------------

    def find_next_pending_job(self) -> Optional[str]:
        """
        Returns the S3 key for the first pending job JSON, or None if no jobs.
        Keys look like: jobs/pending/<job_id>.json
        """
        resp = self.s3.list_objects_v2(
            Bucket=self.cfg.bucket,
            Prefix=self.cfg.pending_prefix,
        )
        contents = resp.get("Contents")
        if not contents:
            return None

        # pick the oldest (by LastModified)
        contents_sorted = sorted(contents, key=lambda o: o["LastModified"])
        for obj in contents_sorted:
            key = obj["Key"]
            if key.endswith(".json"):
                return key

        return None

    # ------ Job JSON ---------------------------------------------------------

    def load_job_json(self, key: str) -> Dict[str, Any]:
        obj = self.s3.get_object(Bucket=self.cfg.bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)

    def delete_key(self, key: str) -> None:
        self.s3.delete_object(Bucket=self.cfg.bucket, Key=key)

    # ------ Status -----------------------------------------------------------

    def _status_key(self, job_id: str) -> str:
        return f"{self.cfg.jobs_prefix}{job_id}/status.json"

    def update_status(self, job_id: str, status: str,
                      message: str = "",
                      progress: int = 0,
                      outputs: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "outputs": outputs or {},
        }
        self.s3.put_object(
            Bucket=self.cfg.bucket,
            Key=self._status_key(job_id),
            Body=json.dumps(payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    # ------ Video I/O --------------------------------------------------------

    def download_video(self, s3_key: str, local_path: str) -> None:
        self.s3.download_file(self.cfg.bucket, s3_key, local_path)

    def upload_video(self, local_path: str, s3_key: str) -> None:
        self.s3.upload_file(local_path, self.cfg.bucket, s3_key)


# -----------------------------------------------------------------------------
# MediaPipe Pose Utilities
# -----------------------------------------------------------------------------

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = list(mp_pose.POSE_CONNECTIONS)

# A subset of key joints for Johansson-style dots
JOHANSSON_LANDMARKS = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]


def _landmark_to_px(landmark, width: int, height: int):
    return int(landmark.x * width), int(landmark.y * height)


# -----------------------------------------------------------------------------
# Rendering functions
# -----------------------------------------------------------------------------

def render_pose_overlay(
    input_path: str,
    output_path: str,
    mode: str = "dots",
    dot_radius: int = 5,
) -> None:
    """
    mode = "dots"      -> Johansson-style dots on joints
    mode = "skeleton"  -> full skeleton lines + joints
    """

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            # start with black background
            black = np.zeros_like(frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if mode == "skeleton":
                    # draw connections
                    for start_idx, end_idx in POSE_CONNECTIONS:
                        p1 = landmarks[start_idx]
                        p2 = landmarks[end_idx]
                        x1, y1 = _landmark_to_px(p1, width, height)
                        x2, y2 = _landmark_to_px(p2, width, height)
                        cv2.line(black, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    # draw all joints as small dots
                    for lm in landmarks:
                        x, y = _landmark_to_px(lm, width, height)
                        cv2.circle(black, (x, y), dot_radius, (255, 255, 255), -1)

                else:
                    # Johansson-like: only important joints, clean dots
                    for lm_id in JOHANSSON_LANDMARKS:
                        lm = landmarks[lm_id]
                        x, y = _landmark_to_px(lm, width, height)
                        cv2.circle(black, (x, y), dot_radius, (255, 255, 255), -1)

            out.write(black)

    cap.release()
    out.release()


# -----------------------------------------------------------------------------
# Main worker loop
# -----------------------------------------------------------------------------

def extract_job_id_from_key(pending_key: str) -> str:
    """
    jobs/pending/20260112_150620__abcd12.json -> 20260112_150620__abcd12
    """
    filename = pending_key.split("/")[-1]
    if filename.endswith(".json"):
        return filename[:-5]
    return filename


def run_worker_loop():
    cfg = WorkerConfig.from_env()
    s3c = S3Client(cfg)

    print("AI People Reader worker started. Waiting for jobs...")

    while True:
        try:
            pending_key = s3c.find_next_pending_job()
            if not pending_key:
                print("No pending jobs. Sleeping 10s...")
                time.sleep(10)
                continue

            job_id = extract_job_id_from_key(pending_key)
            print(f"Picked job: {job_id} ({pending_key})")

            # Load job config
            job_cfg = s3c.load_job_json(pending_key)
            print("Job config:", job_cfg)

            mode = job_cfg.get("mode", "dots")  # "dots" or "skeleton"
            dot_radius = int(job_cfg.get("dot_radius", 5))

            # Input video key:
            # 1) explicitly provided in JSON, or
            # 2) default convention: jobs/<job_id>/input/input.mp4
            input_key = job_cfg.get(
                "input_key",
                f"{cfg.jobs_prefix}{job_id}/input/input.mp4",
            )

            # Output file name depends on mode (to avoid confusion)
            if mode == "skeleton":
                output_filename = "skeleton.mp4"
            else:
                output_filename = "dots.mp4"

            output_key = job_cfg.get(
                "output_key",
                f"{cfg.jobs_prefix}{job_id}/output/{output_filename}",
            )

            # Update status -> processing
            s3c.update_status(
                job_id,
                status="processing",
                message=f"Processing job in mode='{mode}'",
                progress=10,
            )

            # Local temp files
            with tempfile.TemporaryDirectory() as tmpdir:
                local_in = os.path.join(tmpdir, "input.mp4")
                local_out = os.path.join(tmpdir, "output.mp4")

                print(f"Downloading video from s3://{cfg.bucket}/{input_key}")
                s3c.download_video(input_key, local_in)

                # Do rendering
                print(f"Rendering overlay mode={mode}, dot_radius={dot_radius}")
                render_pose_overlay(
                    input_path=local_in,
                    output_path=local_out,
                    mode=mode,
                    dot_radius=dot_radius,
                )

                print(f"Uploading result to s3://{cfg.bucket}/{output_key}")
                s3c.upload_video(local_out, output_key)

            # Job completed
            s3c.update_status(
                job_id,
                status="done",
                message="Completed",
                progress=100,
                outputs={"overlay_key": output_key, "mode": mode},
            )

            # Remove pending json so it won't be picked again
            s3c.delete_key(pending_key)
            print(f"Job {job_id} completed and pending JSON deleted.\n")

        except Exception as e:
            # Try to update status with error (best effort)
            err_msg = str(e)
            print("ERROR processing job:", err_msg)
            try:
                if "job_id" in locals():
                    s3c.update_status(
                        job_id,
                        status="failed",
                        message=err_msg,
                        progress=100,
                    )
            except Exception:
                pass

            # avoid tight error loop
            time.sleep(5)


if __name__ == "__main__":
    run_worker_loop()
