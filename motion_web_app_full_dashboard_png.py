import streamlit as st
import cv2
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except ImportError:
    MEDIAPIPE_AVAILABLE = False
import numpy as np
import pandas as pd
import tempfile
from collections import deque
import os
import re
from datetime import datetime
from pathlib import Path
import smtplib
import ssl
import mimetypes
from email.message import EmailMessage

# Ensure upload directory exists for slip images
UPLOAD_DIR = Path("user_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Backfill Excel from CSV if it exists and Excel file is missing
_csv_path = Path("user_submissions.csv")
_xlsx_path = Path("user_submissions.xlsx")
if _csv_path.exists() and not _xlsx_path.exists():
    try:
        pd.read_csv(_csv_path).to_excel(_xlsx_path, index=False)
    except Exception:
        pass

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def detect_motion_opencv(frame, prev_frame=None):
    """Fallback motion detection using OpenCV when MediaPipe is not available"""
    if prev_frame is None:
        return []
    
    # Convert frames to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate frame difference
    frame_diff = cv2.absdiff(gray, prev_gray)
    
    # Apply threshold to get motion regions
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours of motion
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motions = []
    
    # Analyze motion based on contour properties
    total_motion_area = sum(cv2.contourArea(c) for c in contours)
    
    if total_motion_area > 1000:  # Significant motion detected
        # Get bounding boxes of motion regions
        motion_regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                motion_regions.append((x, y, w, h))
        
        if motion_regions:
            # Analyze motion patterns
            avg_x = np.mean([x for x, y, w, h in motion_regions])
            avg_y = np.mean([y for x, y, w, h in motion_regions])
            
            # Simple motion classification
            if total_motion_area > 5000:
                motions.append("Significant Motion")
            elif total_motion_area > 2000:
                motions.append("Moderate Motion")
            else:
                motions.append("Light Motion")
    
    return motions

def detect_motion_v27(landmarks, prev_landmarks=None):
    """Advanced motion detection using MediaPipe pose landmarks"""
    if not MEDIAPIPE_AVAILABLE:
        return []
    
    motions = []
    get = lambda name: np.array([
        landmarks[name.value].x,
        landmarks[name.value].y,
        landmarks[name.value].z
    ])
    lw, rw = get(mp_pose.PoseLandmark.LEFT_WRIST), get(mp_pose.PoseLandmark.RIGHT_WRIST)
    le, re = get(mp_pose.PoseLandmark.LEFT_ELBOW), get(mp_pose.PoseLandmark.RIGHT_ELBOW)
    ls, rs = get(mp_pose.PoseLandmark.LEFT_SHOULDER), get(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    lh, rh = get(mp_pose.PoseLandmark.LEFT_HIP), get(mp_pose.PoseLandmark.RIGHT_HIP)
    la, ra = get(mp_pose.PoseLandmark.LEFT_ANKLE), get(mp_pose.PoseLandmark.RIGHT_ANKLE)

    shoulder_width = np.linalg.norm(ls[:2] - rs[:2])
    hand_distance = np.linalg.norm(lw[:2] - rw[:2])

    delta_rw = delta_lw = delta_hip = delta_ankle = delta_shoulder = np.zeros(3)
    if prev_landmarks is not None:
        get_prev = lambda name: np.array([
            prev_landmarks[name.value].x,
            prev_landmarks[name.value].y,
            prev_landmarks[name.value].z
        ])
        prev_rw, prev_lw = get_prev(mp_pose.PoseLandmark.RIGHT_WRIST), get_prev(mp_pose.PoseLandmark.LEFT_WRIST)
        prev_lh, prev_rh = get_prev(mp_pose.PoseLandmark.LEFT_HIP), get_prev(mp_pose.PoseLandmark.RIGHT_HIP)
        prev_la, prev_ra = get_prev(mp_pose.PoseLandmark.LEFT_ANKLE), get_prev(mp_pose.PoseLandmark.RIGHT_ANKLE)
        prev_ls, prev_rs = get_prev(mp_pose.PoseLandmark.LEFT_SHOULDER), get_prev(mp_pose.PoseLandmark.RIGHT_SHOULDER)

        delta_rw, delta_lw = rw - prev_rw, lw - prev_lw
        delta_hip = ((lh+rh)/2 - (prev_lh+prev_rh)/2)
        delta_ankle = ((la+ra)/2 - (prev_la+prev_ra)/2)
        delta_shoulder = ((ls+rs)/2 - (prev_ls+prev_rs)/2)

    left_elbow_angle = angle_3pts(ls, le, lw)
    right_elbow_angle = angle_3pts(rs, re, rw)

    # Basic motion detection when MediaPipe is available
    if hand_distance < 0.4*shoulder_width:
        motions.append("Enclosing")
    elif hand_distance > 1.2*shoulder_width:
        motions.append("Spreading")
    
    return motions

# ==== Streamlit Web App ====
# LMA Theme Configuration
st.set_page_config(
    page_title="Movement Matters",
    page_icon="🕴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Movement Matters gray theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #3a3a3a 0%, #2f2f2f 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: #f2f2f2;
        text-align: center;
    }
    .lma-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #e6e6e6;
    }
    .lma-subtitle {
        font-size: 1.1rem;
        color: #bdbdbd;
        font-style: italic;
    }
    .motion-card {
        background: #f5f5f5;
        border-left: 4px solid #7a7a7a;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .metric-box {
        background: #fafafa;
        border: 2px solid #b3b3b3;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(45deg, #6e6e6e, #555555);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #5a5a5a, #444444);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Movement Matters Header
st.markdown("""
<div class="main-header">
    <div class="lma-title">🕴️ Movement Matters</div>
</div>
""", unsafe_allow_html=True)

# Media section: show bundled video and image if available
video_candidates = [
    "vdo present.mp4",
    "present.mp4",
    "present.MP4",
]
media_video = None
for candidate in video_candidates:
    candidate_path = Path(candidate)
    if candidate_path.exists():
        media_video = candidate_path
        break
media_image = Path("picture.jpg")
if (media_video is not None and media_video.exists()) or media_image.exists():
    st.markdown("### 🎥 Media")
    col_v, col_i = st.columns(2)
    with col_v:
        if media_video is not None and media_video.exists():
            st.video(str(media_video))
            st.caption(f"Video: {media_video.name}")
        else:
            st.info("Add file 'vdo present.mp4' or 'present.mp4' to show a video here.")
    with col_i:
        if media_image.exists():
            st.image(str(media_image), caption="picture.jpg", use_container_width=True)
        else:
            st.info("Add file 'picture.jpg' to show an image here.")
else:
    st.markdown("### 🎥 Media")
    st.info("Place 'vdo present.mp4' (or 'present.mp4') and 'picture.jpg' in the app folder to display them here.")

st.markdown("### 📊 **Movement Analysis Dashboard**")
st.markdown("*Upload video for comprehensive motion analysis using Movement Matters principles*")

# --- Sidebar: User Information Form ---
with st.sidebar:
    st.header("User details")
    with st.form("user_details_form", clear_on_submit=False):
        user_name = st.text_input("User name", placeholder="Enter your name")
        user_email = st.text_input("Email", placeholder="name@example.com")
        slip_file = st.file_uploader(
            "Slip (image file)", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="slip_image"
        )
        submitted_user = st.form_submit_button("Submit")

        if submitted_user:
            validation_errors = []
            if not user_name.strip():
                validation_errors.append("User name is required.")
            if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", user_email.strip()):
                validation_errors.append("Valid email is required.")
            if slip_file is None:
                validation_errors.append("Please upload a slip image (PNG/JPG).")

            if validation_errors:
                for err in validation_errors:
                    st.error(err)
            else:
                suffix = Path(slip_file.name).suffix.lower() or ".png"
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", user_name.strip())
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_filename = f"{timestamp_str}_{safe_name}{suffix}"
                saved_path = UPLOAD_DIR / saved_filename
                with open(saved_path, "wb") as f:
                    f.write(slip_file.read())

                csv_path = Path("user_submissions.csv")
                write_header = not csv_path.exists()
                with open(csv_path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write("timestamp,name,email,slip_path\n")
                    f.write(f"{datetime.now().isoformat(timespec='seconds')},{user_name},{user_email},{saved_path.as_posix()}\n")

                # Also write/append to Excel file
                try:
                    xlsx_path = Path("user_submissions.xlsx")
                    new_row = pd.DataFrame([
                        {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "name": user_name,
                            "email": user_email,
                            "slip_path": saved_path.as_posix(),
                        }
                    ])
                    if xlsx_path.exists():
                        try:
                            existing_df = pd.read_excel(xlsx_path)
                            updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                        except Exception:
                            updated_df = new_row
                    else:
                        updated_df = new_row
                    updated_df.to_excel(xlsx_path, index=False)
                except Exception as e:
                    st.warning(f"Saved CSV, but failed saving Excel: {e}")

                st.success("Details saved successfully.")
                st.image(str(saved_path), caption="Uploaded slip", use_container_width=True)

uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","avi"], help="Upload a video file for motion analysis")

if uploaded_file is not None:
    try:
        # Check file size (200MB limit)
        file_size = len(uploaded_file.getvalue())
        if file_size > 200 * 1024 * 1024:  # 200MB
            st.error("File too large! Please upload a file smaller than 200MB.")
        else:
            # Reset file pointer
            uploaded_file.seek(0)
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name

            st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size // (1024*1024)}MB)")
            st.video(video_path)
            
            if st.button("🕴️ **Start Movement Matters Analysis**"):
                st.markdown("## 🔬 **Movement Matters Analysis in Progress**")
                st.markdown("""
                <div class="motion-card">
                    <h4>🔄 Processing Video with Movement Matters Analysis</h4>
                    <p>Analyzing movement patterns, spatial relationships, and temporal dynamics...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                output_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
                output_segment_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
                output_summary_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

                # Initialize pose detection based on availability
                if MEDIAPIPE_AVAILABLE:
                    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
                else:
                    pose = None
                    st.info("Using OpenCV-based motion detection (MediaPipe not available)")

                results_data, motion_segments = [], []
                motion_summary, ongoing = {}, {}
                frame_idx, last_logged_sec = 0, -1
                prev_landmarks = None
                prev_frame = None
                history = deque(maxlen=3)

                fade_text, fade_counter = [], 0
                last_overlay_update_sec = -1

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break

                    motions = []
                    
                    if MEDIAPIPE_AVAILABLE and pose is not None:
                        # Use MediaPipe pose detection
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = pose.process(frame_rgb)

                        if results.pose_landmarks:
                            motions = detect_motion_v27(results.pose_landmarks.landmark, prev_landmarks)
                            prev_landmarks = results.pose_landmarks.landmark

                            # Skeleton overlay red line + white dot
                            mp_drawing.draw_landmarks(
                                frame,
                                results.pose_landmarks,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                            )
                    else:
                        # Use OpenCV fallback motion detection
                        motions = detect_motion_opencv(frame, prev_frame)
                        prev_frame = frame.copy()

                    if motions:
                        history.append(motions)
                        smooth_motions = list({m for hist in history for m in hist})

                        timestamp_sec = frame_idx / fps
                        timestamp_text = f"{int(timestamp_sec//60)}:{int(timestamp_sec%60):02d}"

                        # Overlay text updates only once per second
                        if int(timestamp_sec) != last_overlay_update_sec and smooth_motions:
                            fade_text = smooth_motions[:3]  # max 3 lines
                            fade_counter = int(fps*2)  # 2 sec fade
                            last_overlay_update_sec = int(timestamp_sec)

                        if fade_counter > 0:
                            y_offset = int(height*0.1)
                            for motion in fade_text:
                                cv2.putText(frame, motion, (52, y_offset+2), FONT, FONT_SCALE, (0,0,0), FONT_THICKNESS+1, cv2.LINE_AA)
                                cv2.putText(frame, motion, (50, y_offset), FONT, FONT_SCALE, (255,255,255), FONT_THICKNESS, cv2.LINE_AA)
                                y_offset += 40
                            fade_counter -= 1

                        # 1-sec motion log
                        if smooth_motions and int(timestamp_sec) != last_logged_sec:
                            results_data.append([timestamp_text, ", ".join(smooth_motions)])
                            last_logged_sec = int(timestamp_sec)

                        # Segment tracking (≥0.3s)
                        current_motions = set(smooth_motions)
                        for motion in list(ongoing.keys()):
                            if motion not in current_motions:
                                start_time = ongoing[motion]
                                end_time = timestamp_sec
                                duration = end_time - start_time
                                if duration >= 0.3:
                                    motion_segments.append([motion, start_time, end_time, duration])
                                    motion_summary[motion] = motion_summary.get(motion, 0) + duration
                                del ongoing[motion]
                        for motion in current_motions:
                            if motion not in ongoing:
                                ongoing[motion] = timestamp_sec

                    out.write(frame)
                    frame_idx += 1

                # Close remaining segments
                total_sec = frame_idx/fps
                for motion,start_time in ongoing.items():
                    duration = total_sec - start_time
                    if duration >= 0.3:
                        motion_segments.append([motion, start_time, total_sec, duration])
                        motion_summary[motion] = motion_summary.get(motion, 0) + duration

                cap.release(); out.release()
                if MEDIAPIPE_AVAILABLE and pose is not None:
                    pose.close()

                # Save CSVs with integer values
                df_log = pd.DataFrame(results_data, columns=["timestamp","motions"])
                
                # Convert duration values to integers (seconds)
                motion_segments_int = []
                for segment in motion_segments:
                    motion_segments_int.append([
                        segment[0],  # motion name
                        int(segment[1]),  # start_sec as integer
                        int(segment[2]),  # end_sec as integer
                        int(segment[3])   # duration_sec as integer
                    ])
                
                # Convert summary values to integers
                motion_summary_int = {}
                for motion, duration in motion_summary.items():
                    motion_summary_int[motion] = int(duration)
                
                df_segments = pd.DataFrame(motion_segments_int, columns=["motion","start_sec","end_sec","duration_sec"])
                df_summary = pd.DataFrame(list(motion_summary_int.items()), columns=["motion","total_duration_sec"])

                with open(output_csv,"w") as f: df_log.to_csv(f,index=False)
                with open(output_segment_csv,"w") as f: df_segments.to_csv(f,index=False)
                with open(output_summary_csv,"w") as f: df_summary.to_csv(f,index=False)

                # Movement Matters Styled Results Display
                st.markdown("## 🎯 **Analysis Complete**")
                st.markdown("""
                <div class="motion-card">
                    <h4>✅ Motion Analysis Successfully Completed</h4>
                    <p>Your video has been processed using Movement Matters principles. All motion data has been analyzed and exported.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>📊 Total Motions</h3>
                        <h2>{len(motion_summary_int)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_frames = int(frame_idx)
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>🎬 Frames Analyzed</h3>
                        <h2>{total_frames:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    total_duration = int(frame_idx/fps)
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>⏱️ Video Duration</h3>
                        <h2>{total_duration}s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.video(output_video)
                
                # Movement Matters Styled download buttons
                st.markdown("## 📥 **Download Analysis Reports**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "📋 Download Motion Log CSV", 
                        data=open(output_csv,"rb"), 
                        file_name="movement_matters_motion_log.csv",
                        help="Detailed timestamp-based motion records"
                    )
                    st.download_button(
                        "📊 Download Motion Segments CSV", 
                        data=open(output_segment_csv,"rb"), 
                        file_name="movement_matters_motion_segments.csv",
                        help="Motion segments with start/end times and durations"
                    )
                
                with col2:
                    st.download_button(
                        "📈 Download Motion Summary CSV", 
                        data=open(output_summary_csv,"rb"), 
                        file_name="movement_matters_motion_summary.csv",
                        help="Summary statistics for each motion type"
                    )
                    st.download_button(
                        "🎬 Download Analysis Video", 
                        data=open(output_video,"rb"), 
                        file_name="movement_matters_motion_analysis.mp4",
                        help="Processed video with skeleton overlay and motion labels"
                    )
                
                # Cleanup temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(output_csv)
                    os.unlink(output_segment_csv)
                    os.unlink(output_summary_csv)
                    os.unlink(output_video)
                except:
                    pass
                    
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        st.info("Please try uploading a different video file or check the file format.")
        # Cleanup on error
        try:
            if 'video_path' in locals():
                os.unlink(video_path)
        except:
            pass
