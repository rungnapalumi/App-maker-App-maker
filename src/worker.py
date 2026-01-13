import os
import json
import time
import boto3
import tempfile
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# ---------------------------------------------------------
# 1) LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------
AWS_ACCESS_KEY_ID     = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION            = os.environ.get("AWS_REGION", "ap-southeast-1")
AWS_BUCKET            = os.environ.get("AWS_BUCKET")
JOB_POLL_INTERVAL     = int(os.environ.get("JOB_POLL_INTERVAL", "10"))

# ---------------------------------------------------------
# 2) CREATE S3 CLIENT
# ---------------------------------------------------------
s3 = boto3.client(
    "s3",
    region_name = AWS_REGION,
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

# ---------------------------------------------------------
# 3) IMPORTANT: FIXED MEDIAPIPE IMPORT
# ---------------------------------------------------------
import mediapipe as mp
mp_pose = mp.solutions.pose


# ---------------------------------------------------------
# 4) PROCESS VIDEO → DOT MOTION
# ---------------------------------------------------------
def generate_dot_video(input_path, output_path, dot_size=2):

    print("Processing video:", input_path)

    cap = cv2.VideoCapture(input_path)
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_no_audio = output_path.replace(".mp4", "_temp.mp4")

    writer = cv2.VideoWriter(temp_no_audio, fourcc, fps, (width, height))

    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            out_frame = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(out_frame, (x, y), dot_size, (255, 255, 255), -1)

            writer.write(out_frame)

    cap.release()
    writer.release()

    # ---- Add Audio Back ----------------------------------
    try:
        original = VideoFileClip(input_path)
        processed = VideoFileClip(temp_no_audio)

        if original.audio:
            final = processed.set_audio(original.audio)
        else:
            final = processed

        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )

        original.close()
        processed.close()
        final.close()

    except Exception as e:
        print("Audio merge failed:", e)
        os.rename(temp_no_audio, output_path)

    if os.path.exists(temp_no_audio):
        os.remove(temp_no_audio)

    print("Processing complete:", output_path)


# ---------------------------------------------------------
# 5) PROCESS A SINGLE JOB
# ---------------------------------------------------------
def process_job(job_id, job_key):

    input_key = f"jobs/pending/{job_id}/input/input.mp4"
    output_key = f"jobs/output/{job_id}.json"

    print("Downloading input video from S3:", input_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_input  = os.path.join(tmpdir, "input.mp4")
        local_output = os.path.join(tmpdir, "dots.mp4")
        result_json  = os.path.join(tmpdir, "result.json")

        # --- Download video
        s3.download_file(AWS_BUCKET, input_key, local_input)

        # --- Process video
        generate_dot_video(local_input, local_output, dot_size=2)

        # --- Upload processed video
        s3.upload_file(local_output, AWS_BUCKET, f"jobs/output/{job_id}/dots.mp4")

        result = {"status": "done", "job_id": job_id}
        with open(result_json, "w") as f:
            json.dump(result, f)

        s3.upload_file(result_json, AWS_BUCKET, output_key)

        # --- Clean pending folder
        s3.delete_object(Bucket=AWS_BUCKET, Key=input_key)

    print("Job completed:", job_id)


# ---------------------------------------------------------
# 6) MAIN LOOP
# ---------------------------------------------------------
def main():
    print("Worker started, polling for jobs every", JOB_POLL_INTERVAL, "seconds…")

    while True:
        time.sleep(JOB_POLL_INTERVAL)

        response = s3.list_objects_v2(
            Bucket=AWS_BUCKET,
            Prefix="jobs/pending/",
        )

        if "Contents" not in response:
            continue

        for item in response["Contents"]:
            key = item["Key"]
            if key.endswith(".json"):
                try:
                    job_id = key.split("/")[-1].replace(".json", "")
                    print("Found job:", job_id)
                    process_job(job_id, key)
                except Exception as e:
                    print("FAILED:", job_id, str(e))


if __name__ == "__main__":
    main()
