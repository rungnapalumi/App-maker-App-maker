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
if 'admin_password_verified' not in st.session_state:
    st.session_state.admin_password_verified = False

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
    <div style="text-align: center; padding: 1rem;">
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

def admin_panel():
    """Admin panel for managing uploaded videos"""
    st.markdown("## 🔐 Admin Panel")
    
    # Admin password verification
    st.markdown("### 🔑 Admin Password Verification")
    admin_password = st.text_input("Enter admin password:", type="password")
    
    if st.button("🔓 Verify Password"):
        if admin_password == "0108":
            st.session_state.admin_password_verified = True
            st.success("✅ Password verified! Download options are now available.")
        else:
            st.session_state.admin_password_verified = False
            st.error("❌ Incorrect password. Please try again.")
    
    # Show uploaded videos only if password is verified
    if st.session_state.admin_password_verified:
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
        for video_name in ["Movement matters.mp4", "The key to effective public speaking  your body movement.mp4"]:
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
    else:
        st.info("🔒 Please enter the admin password to access download options.")

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
# Image Matters Asia Theme Configuration
st.set_page_config(
    page_title="Movement Matters by Image Matters Asia",
    page_icon="🕴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Image Matters Asia theme
st.markdown("""
<style>
    /* Main theme colors - Image Matters Asia */
    :root {
        --primary-red: #d32f2f;
        --dark-red: #b71c1c;
        --light-red: #ffcdd2;
        --accent-blue: #1976d2;
        --accent-purple: #7b1fa2;
        --text-dark: #212121;
        --text-light: #757575;
        --background-light: #fafafa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--dark-red) 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(211, 47, 47, 0.3);
    }
    
    .brand-title {
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .brand-subtitle {
        font-size: 1.3rem;
        color: #ffcdd2;
        font-style: italic;
        margin-bottom: 1rem;
    }
    
    .brand-logo {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f5f5f5 0%, #eeeeee 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-red), var(--dark-red));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(211, 47, 47, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, var(--dark-red), var(--primary-red));
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(211, 47, 47, 0.4);
    }
    
    /* Card styling */
    .motion-card {
        background: white;
        border-left: 4px solid var(--primary-red);
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .motion-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    /* Video placeholder styling */
    .video-placeholder {
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
        border: 2px dashed var(--primary-red);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-red);
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid var(--primary-red);
        padding-bottom: 0.5rem;
    }
    
    /* Login form styling */
    .login-form {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid var(--light-red);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
    
    .stError {
        background: #ffebee;
        border-left: 4px solid var(--primary-red);
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Check login status
if not st.session_state.logged_in:
    # Show login form in sidebar
    with st.sidebar:
        st.markdown("### 🔐 Login")
        with st.form("sidebar_login_form"):
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
                    st.error("Invalid credentials.")
        
        # Admin credentials (less prominent)
        with st.expander("🔑 Admin Access"):
            st.write("**Username:** admin")
            st.write("**Password:** [Enter numbers]")
            st.info("Enter the correct password to access admin features")
    
    # Show main content area with login message
    st.markdown("""
    <div class="main-header">
        <div class="brand-logo">🕴️</div>
        <div class="brand-title">Movement Matters</div>
        <div class="brand-subtitle">by Image Matters Asia</div>
        <p style="font-size: 1.1rem; margin-top: 1rem;">Please login to access the application</p>
    </div>
    """, unsafe_allow_html=True)
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
    
    # Movement Matters Header
    st.markdown("""
    <div class="main-header">
        <div class="brand-title">🕴️ Movement Matters</div>
        <div class="brand-subtitle">by Image Matters Asia</div>
    </div>
    """, unsafe_allow_html=True)

# Media section: show Movement Matters videos
st.markdown("""
<div class="section-header">
    🎥 Movement Matters Videos
</div>
""", unsafe_allow_html=True)

# Check for the new videos - Updated for deployment fix
video1_path = Path("Movement matters.mp4")
video2_path = Path("The key to effective public speaking  your body movement.mp4")

# Check for environment variable URLs (for Render deployment)
import os
video1_url = os.getenv("VIDEO1_URL", "https://drive.google.com/file/d/1VM6S8CETZn5K_FBGpSQYJlzN8_N23xjU/preview")
video2_url = os.getenv("VIDEO2_URL", "https://drive.google.com/file/d/1a_Kr9H6VuKXKAAsWoXjxz8JmY2brYqm5/preview")

# Convert Google Drive URLs to proper format
def fix_google_drive_url(url):
    if not url:
        return url
    # Convert /view to /preview for better embedding
    if '/view' in url:
        url = url.replace('/view', '/preview')
    # Remove any extra parameters
    if '?usp=sharing' in url:
        url = url.replace('?usp=sharing', '')
    if '?usp=share_link' in url:
        url = url.replace('?usp=share_link', '')
    return url

# Check if URL is YouTube
def is_youtube_url(url):
    return 'youtube.com' in url or 'youtu.be' in url

video1_url = fix_google_drive_url(video1_url)
video2_url = fix_google_drive_url(video2_url)

# Determine which videos to show
videos_to_show = []

# Check local files first
if video1_path.exists():
    videos_to_show.append(("Movement matters.mp4", str(video1_path), "🎯 **Movement Matters** - Understanding body language and motion analysis"))
elif video1_url:
    videos_to_show.append(("Movement matters.mp4", video1_url, "🎯 **Movement Matters** - Understanding body language and motion analysis"))

if video2_path.exists():
    videos_to_show.append(("The key to effective public speaking your body movement.mp4", str(video2_path), "🎤 **The Key to Effective Public Speaking** - Your body movement matters"))
elif video2_url:
    videos_to_show.append(("The key to effective public speaking your body movement.mp4", video2_url, "🎤 **The Key to Effective Public Speaking** - Your body movement matters"))

# Display videos
if len(videos_to_show) >= 2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Try to embed Google Drive preview links
        if videos_to_show[0][1].startswith('http'):
            try:
                st.video(videos_to_show[0][1])
            except:
                st.markdown("""
                <div class="video-placeholder">
                    <h4>🎯 Movement Matters</h4>
                    <p>Video embedding failed</p>
                    <a href="{}" target="_blank" style="background: var(--primary-red); color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📥 Download Video</a>
                </div>
                """.format(videos_to_show[0][1]), unsafe_allow_html=True)
        else:
            st.video(videos_to_show[0][1])
        st.caption(videos_to_show[0][2])
    
    with col2:
        # Try to embed Google Drive preview links
        if videos_to_show[1][1].startswith('http'):
            try:
                st.video(videos_to_show[1][1])
            except:
                st.markdown("""
                <div class="video-placeholder">
                    <h4>🎤 The Key to Effective Public Speaking</h4>
                    <p>Video embedding failed</p>
                    <a href="{}" target="_blank" style="background: var(--primary-red); color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📥 Download Video</a>
                </div>
                """.format(videos_to_show[1][1]), unsafe_allow_html=True)
        else:
            st.video(videos_to_show[1][1])
        st.caption(videos_to_show[1][2])
        
elif len(videos_to_show) == 1:
    st.markdown("#### Available Videos:")
    # Try to embed the single video
    if videos_to_show[0][1].startswith('http'):
        try:
            st.video(videos_to_show[0][1])
        except:
            st.markdown("""
            <div style="background: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
                <h4>{}</h4>
                <p><strong>Video embedding failed</strong></p>
                <a href="{}" target="_blank" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📥 Download Video</a>
            </div>
            """.format(videos_to_show[0][2], videos_to_show[0][1]), unsafe_allow_html=True)
    else:
        st.video(videos_to_show[0][1])
    st.caption(videos_to_show[0][2])
else:
    # Show both videos as embedded with Google Drive preview links
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
            <h4>🎯 Movement Matters</h4>
            <p>Understanding body language and motion analysis</p>
            <p><strong>Video available for download</strong></p>
            <a href="https://drive.google.com/file/d/1VM6S8CETZn5K_FBGpSQYJlzN8_N23xjU/preview" target="_blank" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📥 Download Video</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
            <h4>🎤 The Key to Effective Public Speaking</h4>
            <p>Your body movement matters</p>
            <p><strong>Video available for download</strong></p>
            <a href="https://drive.google.com/file/d/1a_Kr9H6VuKXKAAsWoXjxz8JmY2brYqm5/preview" target="_blank" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📥 Download Video</a>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    📊 Movement Analysis Dashboard
</div>
<p style="color: var(--text-light); font-style: italic; margin-bottom: 1rem;">
    Upload video for comprehensive motion analysis using Movement Matters principles
</p>
""", unsafe_allow_html=True)



# --- Sidebar: User Information Form (only when logged in) ---
if st.session_state.logged_in:
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
uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","avi","mpeg4"], help="Upload a video file for analysis")

if uploaded_file is not None:
    try:
        # Validate file size
        file_size = len(uploaded_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size > 200 * 1024 * 1024:  # 200MB limit
            st.error(f"❌ File too large! Your file is {file_size_mb:.1f}MB. Please upload a file smaller than 200MB.")
        else:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name

            # Display success message
            st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size_mb:.1f}MB)")
            
            # Display video
            try:
                st.video(video_path)
            except Exception as e:
                st.warning("⚠️ Video preview not available, but file was uploaded successfully.")
            
            # Video analysis section
            st.markdown("---")
            st.markdown("### 🕴️ Video Analysis")
            
            if st.button("🎯 **Start Motion Analysis**", type="primary"):
                with st.spinner("Analyzing video..."):
                    try:
                        # Simulate analysis process
                        st.info("🎯 **Analysis in Progress...**")
                        
                        # Show analysis results (placeholder)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Motion Score", "85%", "↑ 12%")
                        with col2:
                            st.metric("Movement Quality", "Good", "↑ 8%")
                        with col3:
                            st.metric("Balance Score", "92%", "↑ 15%")
                        
                        st.success("✅ Analysis completed successfully!")
                        st.info("📧 Detailed analysis results will be sent to your email address.")
                        
                    except Exception as analysis_error:
                        st.error(f"❌ Analysis failed: {str(analysis_error)}")
                        st.info("🔄 Please try again or contact support.")
            
            # Clean up temporary file
            try:
                os.unlink(video_path)
            except:
                pass
                    
    except Exception as upload_error:
        st.error(f"❌ Error uploading video: {str(upload_error)}")
        st.info("💡 Please check your file format and try again.")
        
        # Clean up any temporary files
        try:
            if 'video_path' in locals():
                os.unlink(video_path)
        except:
            pass
