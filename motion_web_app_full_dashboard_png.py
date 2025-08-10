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

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if 'username' not in st.session_state:
    st.session_state.username = ""

def check_login(username, password):
    """Check if user credentials are valid"""
    # Admin credentials
    if username.lower() == "admin" and password == "0108":
        return True, True  # logged_in, is_admin
    # Regular user (any username with any password for demo)
    elif username and password:
        return True, False  # logged_in, not admin
    return False, False

def login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1>🕴️ Movement Matters</h1>
        <h3>Login Required</h3>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            logged_in, is_admin = check_login(username, password)
            if logged_in:
                st.session_state.logged_in = True
                st.session_state.is_admin = is_admin
                st.session_state.username = username
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <p><strong>Admin Access:</strong> Use username "admin" and password "0108"</p>
        <p><strong>Regular User:</strong> Any username and password combination</p>
    </div>
    """, unsafe_allow_html=True)

def admin_panel():
    """Admin panel for managing uploaded videos"""
    st.markdown("## 🔐 Admin Panel")
    
    # Show uploaded videos
    st.markdown("### 📁 Uploaded Videos")
    
    # Check for uploaded videos in user_uploads directory
    upload_dir = Path("user_uploads")
    video_files = []
    
    if upload_dir.exists():
        for file in upload_dir.glob("*"):
            if file.suffix.lower() in ['.mp4', '.mov', '.avi', '.mpeg4']:
                video_files.append(file)
    
    # Check for existing videos in root directory
    root_videos = []
    for video_name in ["vdo present.mp4", "present.mp4", "present.MP4"]:
        video_path = Path(video_name)
        if video_path.exists():
            root_videos.append(video_path)
    
    if video_files or root_videos:
        st.success(f"Found {len(video_files)} uploaded videos and {len(root_videos)} system videos")
        
        # Display uploaded videos
        if video_files:
            st.markdown("#### 📤 User Uploaded Videos:")
            for i, video_file in enumerate(video_files):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{video_file.name}** ({video_file.stat().st_size // (1024*1024)}MB)")
                with col2:
                    st.video(str(video_file))
                with col3:
                    st.download_button(
                        f"📥 Download {video_file.name}",
                        data=open(video_file, "rb"),
                        file_name=video_file.name,
                        key=f"download_uploaded_{i}"
                    )
        
        # Display system videos
        if root_videos:
            st.markdown("#### 🖥️ System Videos:")
            for i, video_file in enumerate(root_videos):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{video_file.name}** ({video_file.stat().st_size // (1024*1024)}MB)")
                with col2:
                    st.video(str(video_file))
                with col3:
                    st.download_button(
                        f"📥 Download {video_file.name}",
                        data=open(video_file, "rb"),
                        file_name=video_file.name,
                        key=f"download_system_{i}"
                    )
    else:
        st.info("No videos found in the system.")
    
    # Show user submissions
    st.markdown("### 👥 User Submissions")
    csv_path = Path("user_submissions.csv")
    xlsx_path = Path("user_submissions.xlsx")
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df)
        st.download_button(
            "📥 Download User Submissions CSV",
            data=df.to_csv(index=False),
            file_name="user_submissions.csv"
        )
    
    if xlsx_path.exists():
        st.download_button(
            "📥 Download User Submissions Excel",
            data=open(xlsx_path, "rb"),
            file_name="user_submissions.xlsx"
        )

def logout_button():
    """Logout button in sidebar"""
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        st.session_state.username = ""
        st.rerun()

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

# Check login status
if not st.session_state.logged_in:
    login_page()
else:
    # User is logged in - show main app
    # Add logout button to sidebar
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")
    if st.session_state.is_admin:
        st.sidebar.markdown("🔐 **Admin Access**")
    logout_button()
    
    # Show admin panel if admin user
    if st.session_state.is_admin:
        admin_panel()
        st.markdown("---")
    
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
        user_name = st.text_input("Name", placeholder="Name", max_chars=15)
        user_email = st.text_input("Email", placeholder="email@example.com", max_chars=25)
        slip_file = st.file_uploader(
            "Slip", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="slip_image"
        )
        submitted_user = st.form_submit_button("Submit")

        if submitted_user:
            if user_name.strip() and user_email.strip() and slip_file is not None:
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
                except Exception:
                    pass

                st.success("Saved!")
                st.image(str(saved_path), caption="Uploaded slip", use_container_width=True)

# Video Upload Section (Working normally)
uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","avi"], help="Upload a video file")

if uploaded_file is not None:
    try:
        file_size = len(uploaded_file.getvalue())
        if file_size > 200 * 1024 * 1024:  # 200MB
            st.error("File too large! Please upload a file smaller than 200MB.")
        else:
            uploaded_file.seek(0)
            
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            video_path = tfile.name

            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.video(video_path)
            
            # Video analysis button (hidden functionality)
            if st.button("🕴️ **Start Analysis**"):
                st.info("🎯 Video analysis feature coming soon!")
                    
    except Exception:
        st.error("Error uploading video. Please try again.")
        try:
            if 'video_path' in locals():
                os.unlink(video_path)
        except:
            pass
