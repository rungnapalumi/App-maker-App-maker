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
            st.write("**Password:** [Hidden for security]")
            st.info("Click to copy credentials")
    
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
video1_url = os.getenv("VIDEO1_URL", "")
video2_url = os.getenv("VIDEO2_URL", "")

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
        # Only embed YouTube videos
        if videos_to_show[0][1].startswith('http') and is_youtube_url(videos_to_show[0][1]):
            st.video(videos_to_show[0][1])
        else:
            st.markdown("""
            <div class="video-placeholder">
                <h4>🎯 Movement Matters</h4>
                <p>Video will be embedded here</p>
                <p><em>Upload to YouTube for embedded playback</em></p>
            </div>
            """, unsafe_allow_html=True)
        st.caption(videos_to_show[0][2])
    
    with col2:
        # Only embed YouTube videos
        if videos_to_show[1][1].startswith('http') and is_youtube_url(videos_to_show[1][1]):
            st.video(videos_to_show[1][1])
        else:
            st.markdown("""
            <div class="video-placeholder">
                <h4>🎤 The Key to Effective Public Speaking</h4>
                <p>Video will be embedded here</p>
                <p><em>Upload to YouTube for embedded playback</em></p>
            </div>
            """, unsafe_allow_html=True)
        st.caption(videos_to_show[1][2])
        
elif len(videos_to_show) == 1:
    st.markdown("#### Available Videos:")
    # Always show info card for single video
    st.markdown("""
    <div style="background: #f0f0f0; padding: 20px; border-radius: 10px; text-align: center;">
        <h4>{}</h4>
        <p><strong>Video available for download</strong></p>
        <a href="{}" target="_blank" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">📥 Download Video</a>
    </div>
    """.format(videos_to_show[0][2], videos_to_show[0][1] if videos_to_show[0][1].startswith('http') else "https://drive.google.com/file/d/1VM6S8CETZn5K_FBGpSQYJlzN8_N23xjU/preview"), unsafe_allow_html=True)
    st.caption(videos_to_show[0][2])
else:
    # Show both videos as info cards even if not found
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

# --- Official Movement Matters Content from Image Matters Asia ---
st.markdown("""
<div class="motion-card">
    <h2 style="color: var(--primary-red); text-align: center; margin-bottom: 2rem;">MOVEMENT MATTERS</h2>
    <p style="font-size: 1.2rem; font-style: italic; text-align: center; margin-bottom: 2rem;">
        <strong>Body Movement is a psycho-physical process - an outward expression of inner intent.</strong><br>
        The use of congruent body movement in combination with the verbal message can enhance communication.
    </p>
</div>
""", unsafe_allow_html=True)

# What people don't consider when presenting
st.markdown("""
<div class="motion-card">
    <h3 style="color: var(--primary-red); margin-bottom: 1rem;">What people don't consider when presenting:</h3>
    <ul style="margin-left: 1rem;">
        <li><strong>What others see.</strong></li>
        <li><strong>One's signature body movement</strong> or repetitive movement which could interfere with the verbal message.</li>
        <li><strong>Shape:</strong> Using different shapes of movement in the following planes:
            <ul style="margin-left: 1rem; margin-top: 0.5rem;">
                <li><strong>Horizontal</strong> = communicating with spreading and enclosing</li>
                <li><strong>Vertical</strong> = sense of authority through the weight effort giving what is presented "weight", literally….expressing a determination or resolve, an ability to argue for a particular aspects over others.</li>
                <li><strong>Sagittal Planes</strong> = going forward into the future and going backwards reflecting on the past.</li>
            </ul>
        </li>
        <li><strong>One's overall body attitude or static body posture:</strong>
            <ul style="margin-left: 1rem; margin-top: 0.5rem;">
                <li><strong>Pin Body Attitude</strong> - communicates rigidity.</li>
                <li><strong>Wall Body Attitude</strong> – communicates being one-dimensional and appearing unapproachable and authoritative.</li>
                <li><strong>Ball Body Attitude</strong> - communicates caving in, and being humble and sometimes weakness.</li>
                <li><strong>Screw Body Attitude</strong> - communicates approachability, ability to use different dimensions, and ability to do complex tasks.</li>
            </ul>
        </li>
        <li><strong>The use of one's reach and hand gesture</strong>, for example, using near, mid or far reach.</li>
        <li><strong>Whether one's movement involves whole or just part of body.</strong></li>
        <li><strong>One's "Phrasing":</strong>
            <ul style="margin-left: 1rem; margin-top: 0.5rem;">
                <li><strong>Even Phrasing</strong> – communicates consistency and calmness.</li>
                <li><strong>Impulsive Phrasing</strong> – communicates abruptness, urgency, chaotic, rushing qualities with no preparation.</li>
                <li><strong>Impactive Phrasing</strong> – communicates weight and solidity, insistent, clear, declarative.</li>
                <li><strong>Swing Phrasing</strong> – communicates resiliency, cooperative, easy going, playful.</li>
                <li><strong>Vibratory Phrasing</strong> – communicates nervousness, hectic, chaotic and out of control qualities.</li>
            </ul>
        </li>
        <li><strong>Space:</strong> whether movement is focused or scattered.</li>
        <li><strong>Weight:</strong> whether movement is with pressure, force or with sensitivity.</li>
        <li><strong>Time:</strong> whether movement is with speed, sudden expressing sense of urgency or taking time slowing of the pace.</li>
        <li><strong>PGMs:</strong> the use of posture and gesture mergers to create genuine and conflict-free expressions.</li>
        <li><strong>Expanding one's own scope of movement possibilities.</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Movement in Communication
st.markdown("""
<div class="motion-card">
    <h3 style="color: var(--primary-red); margin-bottom: 1rem;">Movement in Communication</h3>
    <p style="font-style: italic; margin-bottom: 1rem;">Alisa on "Movement in Communication", Heidelberg University, Germany</p>
</div>
""", unsafe_allow_html=True)

# What You Will Learn
st.markdown("""
<div class="motion-card">
    <h3 style="color: var(--primary-red); text-align: center; margin-bottom: 2rem;">WHAT YOU WILL LEARN</h3>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 2rem;">
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Your Stance:</h4>
            <p>How wide should your legs and feet be when standing?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Your Body Attitude (Body Posture):</h4>
            <p>Yes, there is such a thing as "Body Attitude"!!! How much do you know about your Body Attitude and how others perceived of you?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Your Head Gesture:</h4>
            <p>How much should you use your hand gesture to ensure they are in congruent with the whole body movement and message?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Your Reach:</h4>
            <p>Should you use near reach or gesture all the time or do you realise that you can also use mid and far reach as well?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Your Feet:</h4>
            <p>Should they be firmly planted or are you planning to move?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Weight Shifting:</h4>
            <p>How do you know when to weight shift and which direction should you go?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Speed:</h4>
            <p>How do you know you're not moving too much, too little, or too abruptly?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Walking:</h4>
            <p>When should you walk and how far should you go? How fast or slow should you walk?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Audience Engagement:</h4>
            <p>How do you engage with your audience?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Weight Effort:</h4>
            <p>How about the weight you use in the delivery of your message?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Upper Body:</h4>
            <p>What about the rotation of your upper body in order not to appear static?</p>
        </div>
        
        <div style="background: var(--light-red); padding: 1rem; border-radius: 8px;">
            <h4 style="color: var(--primary-red); margin-bottom: 0.5rem;">Eye Contact:</h4>
            <p>What sort eye contact should you give your audience?</p>
        </div>
    </div>
    
    <p style="text-align: center; font-size: 1.1rem; font-weight: bold; color: var(--primary-red);">
        And much more….
    </p>
</div>
""", unsafe_allow_html=True)

# Testimonial
st.markdown("""
<div class="motion-card">
    <blockquote style="border-left: 4px solid var(--primary-red); padding-left: 1rem; font-style: italic; margin: 1rem 0;">
        "Understanding that people express feelings through their movements is an essential skill. Alisa has taught me to read people's body language which continues to help me daily in my work in sales. In negotiations I am now able to sense what is going on in the minds of my business partners. But communication always goes two-ways. Alisa is so skilled in her area; she has even profiled my own body movement so I know how I come across towards other people!"
    </blockquote>
    <div style="text-align: right; margin-top: 1rem;">
        <p style="font-weight: bold; margin-bottom: 0;">Bernhard Vreden, M.Sc.</p>
        <p style="color: var(--text-light); margin-bottom: 0;">Key Account Manager</p>
        <p style="color: var(--text-light);">Bodo Möller Chemie GmbH</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Contact Information
st.markdown("""
<div class="motion-card">
    <h3 style="color: var(--primary-red); text-align: center; margin-bottom: 1rem;">Contact Us</h3>
    <div style="text-align: center;">
        <p><strong>Call Us:</strong></p>
        <p><strong>(Austria):</strong> +43 664 6680199</p>
        <p><strong>(Bangkok):</strong> +66 81-357-2315</p>
        <p style="margin-top: 1rem;"><em>Body Stories GmbH, Vienna, Austria</em></p>
    </div>
</div>
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
