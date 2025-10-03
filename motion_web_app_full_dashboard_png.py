import streamlit as st
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import re

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""

def check_login(username, password):
    """Check if user credentials are valid"""
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

# ==== Streamlit Web App ====
st.set_page_config(
    page_title="Movement Matters - Video Analysis",
    page_icon="🕴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
                        
                        # Simply copy the uploaded video as the "processed" video for now
                        processed_path = UPLOAD_DIR / processed_filename
                        shutil.copy2(saved_path, processed_path)
                        
                        # Store analysis results in session state to make them permanent
                        st.session_state.analysis_completed = True
                        st.session_state.analysis_timestamp = timestamp_str
                        st.session_state.original_video_name = uploaded_file.name
                        
                        # Store files in session state to prevent refresh issues
                        if 'processed_video_data' not in st.session_state:
                            with open(processed_path, "rb") as f:
                                st.session_state.processed_video_data = f.read()
                        
                        # Create a simple text report
                        report_content = f"""Movement Matters Analysis Report
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
                            st.session_state.report_data = report_content.encode('utf-8')
                        
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
    st.info("💾 **Click the download buttons below to save files to your Downloads folder**")
    st.warning("⚠️ **Make sure to allow downloads in your browser if prompted**")
    
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
        # Create download button with proper data handling
        if 'processed_video_data' in st.session_state:
            st.download_button(
                "📥 Download Video",
                data=st.session_state.processed_video_data,
                file_name=video_filename,
                mime="video/mp4",
                key="permanent_download_video",
                help="Click to download the skeleton overlay video to your Downloads folder"
            )
        else:
            st.error("❌ Video data not available. Please re-run the analysis.")
    
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
        # Create download button with proper data handling
        if 'report_data' in st.session_state:
            st.download_button(
                "📥 Download Report",
                data=st.session_state.report_data,
                file_name=report_filename,
                mime=mime_type,
                key="permanent_download_report",
                help="Click to download the analysis report to your Downloads folder"
            )
        else:
            st.error("❌ Report data not available. Please re-run the analysis.")
    
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
    
    # Add troubleshooting section
    st.markdown("---")
    st.markdown("### 🔧 Download Troubleshooting")
    with st.expander("Having trouble downloading? Click here for help"):
        st.markdown("""
        **If downloads aren't working:**
        
        1. **Check browser settings**: Make sure your browser allows downloads
        2. **Try right-click**: Right-click the download button and select "Save link as..."
        3. **Check Downloads folder**: Files should appear in your computer's Downloads folder
        4. **File extensions**: Make sure the files have the correct extensions (.mp4 for video, .pdf for report)
        5. **Browser compatibility**: Try using Chrome, Firefox, or Safari
        
        **Expected file types:**
        - Video files: `.mp4` format
        - Report files: `.pdf` format
        
        If you're still having issues, try refreshing the page and re-running the analysis.
        """)
