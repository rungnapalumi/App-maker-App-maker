import streamlit as st
import cv2
import pandas as pd
import mediapipe as mp
import tempfile

st.set_page_config(page_title="Lumi Skeleton Overlay", layout="wide")

st.title("🌸 Skeleton Overlay with Reference Timestamp 💚")
st.write("Upload video + reference CSV → Overlay skeleton & motion text based on CSV timestamps.")

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

uploaded_video = st.file_uploader("Upload a video", type=["mp4","mov","avi"])
uploaded_csv = st.file_uploader("Upload reference CSV", type=["csv"])

if uploaded_video and uploaded_csv:
    # Load CSV
    motion_df = pd.read_csv(uploaded_csv)

    # Ensure timestamp column exists
    if "timestamp" not in motion_df.columns:
        motion_df.rename(columns={motion_df.columns[0]: "timestamp"}, inplace=True)

    # Convert timestamp to seconds
    def time_to_sec(t):
        m, s = t.split(':')
        return int(m)*60 + int(s)
    motion_df['time_sec'] = motion_df['timestamp'].apply(time_to_sec)

    # Determine motion columns (all except timestamp + time_sec)
    motion_cols = [c for c in motion_df.columns if c not in ['timestamp','time_sec']]

    # Prepare video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    st.video(video_path)

    if st.button("Generate Skeleton Overlay"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        font_scale = 0.35
        font_thickness = 1
        margin_x = 20
        margin_y = 20
        frame_idx = 0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    # Draw skeleton: red line + white dot (small)
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=1),  # Red line
                        mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)  # White dot
                    )

                # Current second
                current_sec = int(frame_idx / fps)
                timestamp_text = f"{int(current_sec//60)}:{int(current_sec%60):02d}"

                # Lookup motion from reference CSV
                row = motion_df[motion_df['time_sec'] == current_sec]
                if not row.empty:
                    motions = [col for col in motion_cols if row.iloc[0][col] == 1]
                    if motions:
                        text = " + ".join(motions)

                        # Calculate position bottom-right
                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                        text_x = width - text_size[0] - margin_x
                        text_y = height - margin_y

                        # Outline black
                        cv2.putText(frame, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness+2, cv2.LINE_AA)
                        # White text
                        cv2.putText(frame, text, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

                # Timestamp above motion text
                ts_text = f"Time: {timestamp_text}"
                ts_size, _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                ts_x = width - ts_size[0] - margin_x
                ts_y = height - margin_y - 20
                cv2.putText(frame, ts_text, (ts_x, ts_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

                out.write(frame)
                frame_idx += 1

        cap.release()
        out.release()

        st.success("✅ Skeleton overlay video generated!")
        st.video(output_video)
        st.download_button("Download Motion Overlay Video", data=open(output_video,"rb"), file_name="skeleton_overlay.mp4")
