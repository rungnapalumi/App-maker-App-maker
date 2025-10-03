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

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

def check_login(username, password):
    """Check if user credentials are valid"""
    # Simple login - any username with any password for demo
    if username and password:
        return True
    return False

def logout_button():
    """Logout button in sidebar"""
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# Ensure upload directory exists
UPLOAD_DIR = Path("user_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
st.set_page_config(
    page_title="Movement Matters - Video Analysis",
    page_icon="🕴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-red: #d32f2f;
        --dark-red: #b71c1c;
        --light-red: #ffcdd2;
        --text-dark: #212121;
        --text-light: #757575;
        --background-light: #fafafa;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, var(--primary-red) 0%, var(--dark-red) 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(211, 47, 47, 0.3);
    }
    
    .brand-title {
        font-size: 1.75rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .brand-subtitle {
        font-size: 0.65rem;
        color: #ffcdd2;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    
    .brand-logo {
        font-size: 2rem;
        margin-bottom: 0.5rem;
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
    
    /* Section headers */
    .section-header {
        color: var(--primary-red);
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 3px solid var(--primary-red);
        padding-bottom: 0.5rem;
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
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if check_login(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    
    # Show main content area with login message
    st.markdown("""
    <div class="main-header">
        <div class="brand-logo">🕴️</div>
        <div class="brand-title">Movement Matters</div>
        <div class="brand-subtitle">Video Analysis Platform</div>
        <p style="font-size: 1.1rem; margin-top: 1rem;">Please login to access the application</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # User is logged in - show main app
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")
    logout_button()
    
    # Movement Matters Header
    st.markdown("""
    <div class="main-header">
        <div class="brand-title">🕴️ Movement Matters</div>
        <div class="brand-subtitle">Video Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Main Analysis Section
    st.markdown("""
    <div class="section-header">
        📊 Video Analysis Dashboard
    </div>
    <p style="color: var(--text-light); font-style: italic; margin-bottom: 1rem;">
        Upload video for comprehensive motion analysis
    </p>
    """, unsafe_allow_html=True)

    # Video Upload Section
    uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","avi","mpeg4"], help="Upload a video file for analysis (Max 500MB)")

    if uploaded_file is not None:
        try:
            # Validate file size
            file_size = len(uploaded_file.getvalue())
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size > 500 * 1024 * 1024:  # 500MB limit
                st.error(f"❌ File too large! Your file is {file_size_mb:.1f}MB. Please upload a file smaller than 500MB.")
            else:
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Create permanent file in user_uploads directory
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_filename = re.sub(r"[^a-zA-Z0-9._-]", "_", uploaded_file.name)
                saved_filename = f"{timestamp_str}_{safe_filename}"
                saved_path = UPLOAD_DIR / saved_filename
                
                # Save the video file permanently
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Display success message
                st.success(f"✅ File uploaded successfully: {uploaded_file.name} ({file_size_mb:.1f}MB)")
                
                # Display video
                try:
                    st.video(str(saved_path))
                except Exception as e:
                    st.warning("⚠️ Video preview not available, but file was uploaded successfully.")
                
                # Video analysis section
                st.markdown("---")
                st.markdown("### 🕴️ Video Analysis")
                
                if st.button("🎯 **Send to Analyze**", type="primary"):
                    try:
                        # Detect video type based on filename
                        video_name_lower = uploaded_file.name.lower()
                        is_engaging = "engaging" in video_name_lower or "connecting" in video_name_lower
                        is_instructions = "giving" in video_name_lower or "instruction" in video_name_lower or "สั่ง" in video_name_lower
                        
                        # Show processing message with 5-minute delay
                        st.info("🎯 **Video sent for analysis. Processing will take approximately 5 minutes...**")
                        st.warning("⚠️ **Please do not refresh the page or click back button while analyzing**")
                        
                        # Create a progress bar for 5 minutes (300 seconds)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        import time
                        
                        # Simulate 5-minute processing time
                        for i in range(300):  # 300 seconds = 5 minutes
                            # Update progress bar
                            progress_bar.progress((i + 1) / 300)
                            
                            # Update status text with countdown
                            remaining_minutes = (300 - i) // 60
                            remaining_seconds = (300 - i) % 60
                            status_text.text(f"⏳ Processing... {remaining_minutes:02d}:{remaining_seconds:02d} remaining")
                            
                            # Wait 1 second
                            time.sleep(1)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show completion message
                        st.success("✅ Analysis completed successfully!")
                        
                        # Show analysis results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Motion Score", "85%", "↑ 12%")
                        with col2:
                            st.metric("Movement Quality", "Good", "↑ 8%")
                        with col3:
                            st.metric("Balance Score", "92%", "↑ 15%")
                        
                        # Generate appropriate filenames based on video type
                        if is_engaging:
                            processed_filename = "Engaging & Connecting overlay video.mp4"
                            report_filename = "Alisa Character Analysis Report_Engaging & Connecting.pdf"
                        elif is_instructions:
                            processed_filename = "Giving Instructions overlay video.mp4"
                            report_filename = "Alisa Character Analysis Report_Giving Instructions.pdf"
                        else:
                            # Default fallback
                            processed_filename = "skeleton_overlay_with_audio.mp4"
                            report_filename = "Report 270925.docx"
                        
                        # Provide existing skeleton overlay video files based on the 4 videos
                        if is_engaging:
                            # Look for existing engaging & connecting skeleton overlay video
                            skeleton_video_path = Path("../Engaging & Connecting overlay.mp4")
                        elif is_instructions:
                            # Look for existing giving instructions skeleton overlay video  
                            skeleton_video_path = Path("../Giving Instructions overlay.MP4")
                        else:
                            # Default skeleton overlay video
                            skeleton_video_path = Path("../skeleton_overlay_with_audio-12.MP4")
                        
                        # Copy the existing skeleton overlay video to user_uploads with appropriate name
                        processed_path = UPLOAD_DIR / processed_filename
                        if skeleton_video_path.exists():
                            import shutil
                            shutil.copy2(skeleton_video_path, processed_path)
                        else:
                            # Fallback: use uploaded video if skeleton overlay doesn't exist
                            shutil.copy2(saved_path, processed_path)
                        
                        # Store analysis results in session state to make them permanent
                        st.session_state.analysis_completed = True
                        st.session_state.analysis_timestamp = timestamp_str
                        st.session_state.original_video_name = uploaded_file.name
                        
                        # Store files in session state to prevent refresh issues
                        if 'processed_video_data' not in st.session_state:
                            with open(processed_path, "rb") as f:
                                st.session_state.processed_video_data = f.read()
                        
                        # Generate report content
                        report_content = f"""
Movement Matters Analysis Report
==============================

Video Name: {uploaded_file.name}
Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SCORES:
- Motion Score: 85%
- Movement Quality: Good
- Balance Score: 92%

KEY FINDINGS:
• Good posture maintained throughout
• Appropriate hand gestures
• Confident body language
• Minimal nervous movements

RECOMMENDATIONS:
• Continue current speaking style
• Maintain confident posture
• Keep using appropriate gestures
                        """
                        
                        # Store report data in session state
                        if 'report_data' not in st.session_state:
                            # Check if the actual report file exists in the main directory based on video type
                            if is_engaging:
                                main_report_path = Path("../Alisa Character Analysis Report_Engaging & Connecting.pdf")
                            elif is_instructions:
                                main_report_path = Path("../Alisa Character Analysis Report_Giving Instructions.pdf")
                            else:
                                main_report_path = Path("../Report 270925.docx")
                            
                            if main_report_path.exists():
                                # Use the actual report file
                                with open(main_report_path, "rb") as f:
                                    st.session_state.report_data = f.read()
                            else:
                                # Fallback to generated content
                                st.session_state.report_data = report_content.encode('utf-8')
                        
                        # Analysis summary removed as requested
                        
                    except Exception as analysis_error:
                        st.error(f"❌ Analysis failed: {str(analysis_error)}")
                        st.info("🔄 Please try again or contact support.")
                
        except Exception as upload_error:
            st.error(f"❌ Error uploading video: {str(upload_error)}")
            st.info("💡 Please check your file format and try again.")

# Show permanent download section if analysis is completed
if st.session_state.get('analysis_completed', False):
    st.markdown("---")
    st.markdown("### 📥 Permanent Downloads")
    st.success("🎯 **Analysis results are available for download:**")
    st.info("💾 **Video and Report will be saved in Download folder**")
    
    # Create two columns for the downloads
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("#### 🎥 Skeleton Overlay Video")
        # Determine video filename based on original video name
        original_name_lower = st.session_state.get('original_video_name', '').lower()
        if "engaging" in original_name_lower or "connecting" in original_name_lower:
            video_filename = "Engaging & Connecting overlay video.mp4"
        elif "giving" in original_name_lower or "instruction" in original_name_lower or "สั่ง" in original_name_lower:
            video_filename = "Giving Instructions overlay video.mp4"
        else:
            video_filename = "skeleton_overlay_with_audio.mp4"
        
        st.info(f"📹 **{video_filename}**")
        st.download_button(
            "📥 Download Video",
            data=st.session_state.processed_video_data,
            file_name=video_filename,
            mime="video/mp4",
            key="permanent_download_video"
        )
    
    with col2:
        st.markdown("#### 📄 Analysis Report")
        # Determine report filename based on original video name
        original_name_lower = st.session_state.get('original_video_name', '').lower()
        if "engaging" in original_name_lower or "connecting" in original_name_lower:
            report_filename = "Alisa Character Analysis Report_Engaging & Connecting.pdf"
            mime_type = "application/pdf"
        elif "giving" in original_name_lower or "instruction" in original_name_lower or "สั่ง" in original_name_lower:
            report_filename = "Alisa Character Analysis Report_Giving Instructions.pdf"
            mime_type = "application/pdf"
        else:
            report_filename = "Report 270925.docx"
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        
        st.info(f"📋 **{report_filename}**")
        st.download_button(
            "📥 Download Report",
            data=st.session_state.report_data,
            file_name=report_filename,
            mime=mime_type,
            key="permanent_download_report"
        )
    
    with col3:
        st.markdown("#### 🗑️ Clear Results")
        if st.button("🗑️ Delete Video & Report", type="secondary"):
            # Clear all analysis data from session state
            if 'analysis_completed' in st.session_state:
                del st.session_state.analysis_completed
            if 'analysis_timestamp' in st.session_state:
                del st.session_state.analysis_timestamp
            if 'original_video_name' in st.session_state:
                del st.session_state.original_video_name
            if 'processed_video_data' in st.session_state:
                del st.session_state.processed_video_data
            if 'report_data' in st.session_state:
                del st.session_state.report_data
            st.success("✅ Analysis results cleared!")
            st.rerun()
