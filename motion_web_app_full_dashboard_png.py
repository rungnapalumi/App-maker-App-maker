import streamlit as st
import cv2
import mediapipe as mp
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


def _resolve_smtp_settings() -> dict:
    """Return SMTP settings from st.secrets or environment variables.
    Keys: host, port, user, password, from_addr, to_addr, use_ssl
    """
    smtp_secrets = {}
    try:
        smtp_secrets = st.secrets.get("smtp", {})  # type: ignore[attr-defined]
    except Exception:
        smtp_secrets = {}

    host = smtp_secrets.get("host") or os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(smtp_secrets.get("port") or os.environ.get("SMTP_PORT", "587"))
    user = smtp_secrets.get("user") or os.environ.get("SMTP_USER")
    password = smtp_secrets.get("pass") or os.environ.get("SMTP_PASS")
    from_addr = smtp_secrets.get("from") or os.environ.get("EMAIL_FROM", user or "")
    to_addr = smtp_secrets.get("to") or os.environ.get("EMAIL_TO", "petchpat@gmail.com")
    use_ssl = smtp_secrets.get("ssl")
    if use_ssl is None:
        use_ssl_env = (os.environ.get("SMTP_SSL", "false") or "").strip().lower()
        use_ssl = use_ssl_env in ("1", "true", "yes") or port == 465

    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "from_addr": from_addr,
        "to_addr": to_addr,
        "use_ssl": use_ssl,
    }


def send_submission_email(user_name: str, user_email: str, slip_path: Path) -> tuple[bool, str]:
    """Send an email with the user's details and attach the slip image.

    Reads SMTP settings via _resolve_smtp_settings().
    """
    cfg = _resolve_smtp_settings()
    host = cfg["host"]
    port = cfg["port"]
    smtp_user = cfg["user"]
    smtp_pass = cfg["password"]
    from_addr = cfg["from_addr"]
    to_addr = cfg["to_addr"]
    use_ssl = cfg["use_ssl"]

    if not smtp_user or not smtp_pass or not from_addr:
        return False, "Missing SMTP credentials (user/pass/from). Configure st.secrets['smtp'] or env vars."

    try:
        msg = EmailMessage()
        msg_subject_placeholder = "Movement Matters"  # no-op for context
        msg["Subject"] = f"{msg_subject_placeholder} submission from {user_name}"
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(
            f"New submission received.\n\n"
            f"Name: {user_name}\n"
            f"Email: {user_email}\n"
            f"Slip path: {slip_path.as_posix()}\n"
            f"Submitted at: {datetime.now().isoformat(timespec='seconds')}\n"
        )

        # Attach slip if accessible
        try:
            with open(slip_path, "rb") as f:
                data = f.read()
            ctype, _ = mimetypes.guess_type(str(slip_path))
            if not ctype:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=slip_path.name)
        except Exception:
            pass

        context = ssl.create_default_context()
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, context=context, timeout=20) as server:
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.ehlo()
                try:
                    server.starttls(context=context)
                except Exception:
                    pass
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def angle_3pts(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def detect_motion_v27(landmarks, prev_landmarks=None):
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

    # ---- Motion Rules Based on motion_definitions.py ----
    
    # 1. ADVANCING: "Forward", "Step or lean forward; torso, pelvis", "Straight, linear", "Moderate to fast"
    # "Torso tilts forward, weight on front foot"
    dz_hip, dz_ankle, dz_shoulder = delta_hip[2], delta_ankle[2], delta_shoulder[2]
    dz_forward = dz_hip < -0.0015 or dz_ankle < -0.0015 or dz_shoulder < -0.0015
    if dz_forward:
        motions.append("Advancing")

    # 2. RETREATING: "Backward", "Step or lean backward; torso, pelvis", "Straight, linear", "Moderate to slow"
    # "Torso leans back, arms withdraw"
    dz_backward = dz_hip > 0.0015 or dz_ankle > 0.0015 or dz_shoulder > 0.0015
    if dz_backward:
        motions.append("Retreating")

    # 3. ENCLOSING: "Inward toward midline", "Arms fold inward; shoulders, hands", "Curved inward", "Smooth and continuous"
    # "Shoulders close, hands overlap or contain"
    if hand_distance < 0.4*shoulder_width:
        motions.append("Enclosing")

    # 4. SPREADING: "Outward from center", "Arms extend outward; chest opens", "Curved or straight outward", "Quick-expanding"
    # "Chest lifts, arms and fingers splay"
    if hand_distance > 1.2*shoulder_width:
        motions.append("Spreading")

    # 5. DIRECTING: "Straight toward target", "Pointing/reaching with one joint chain", "Linear and focused", "Sustained or quick"
    # "Eyes and head align with hand/target"
    left_arm_extended = (left_elbow_angle > 150 and lw[2] < ls[2] - 0.1)
    right_arm_extended = (right_elbow_angle > 150 and rw[2] < rs[2] - 0.1)
    if (left_arm_extended and not right_arm_extended) or (right_arm_extended and not left_arm_extended):
        pointing_magnitude = np.linalg.norm(delta_lw[:2] if left_arm_extended else delta_rw[:2])
        if pointing_magnitude > 0.01 and pointing_magnitude < 0.05:
            motions.append("Directing")

    # 6. GLIDING: "Smooth directional path, often forward or sideward", "Arms and hands lead; torso steady", "Linear, smooth", "Sustained"
    # "No abrupt stops, continuous light contact"
    left_arm_movement = np.linalg.norm(delta_lw[:2])
    right_arm_movement = np.linalg.norm(delta_rw[:2])
    if (left_arm_movement > 0.005 and left_arm_movement < 0.03) or (right_arm_movement > 0.005 and right_arm_movement < 0.03):
        if not any(motion in motions for motion in ["Punching", "Pressing", "Dabbing"]):
            if (abs(delta_lw[0]) > 0.003 or abs(delta_rw[0]) > 0.003) or (delta_lw[2] < -0.003 or delta_rw[2] < -0.003):
                motions.append("Gliding")

    # 7. PUNCHING: "Forward or downward in a forceful line", "Whole arm or body used in a direct forceful push", "Straight, heavy trajectory", "Sudden"
    # "Strong muscle engagement, full stop at end"
    if ((right_elbow_angle > 140 or left_elbow_angle > 140) and
        ((delta_rw[2] < -0.005 or delta_lw[2] < -0.005) or
         (abs(delta_rw[0]) > 0.2*shoulder_width or abs(delta_lw[0]) > 0.2*shoulder_width))):
        motions.append("Punching")

    # 8. DABBING: "Short, precise directional path", "Hand and fingers dart with control; wrist involved", "Small, straight path", "Sudden"
    # "Quick precision, like tapping or striking lightly"
    left_dab_magnitude = np.linalg.norm(delta_lw[:2])
    right_dab_magnitude = np.linalg.norm(delta_rw[:2])
    if (left_dab_magnitude > 0.01 and left_dab_magnitude < 0.04) or (right_dab_magnitude > 0.01 and right_dab_magnitude < 0.04):
        if not any(motion in motions for motion in ["Gliding", "Pressing", "Punching"]):
            if (abs(delta_lw[0]) > 0.005 or abs(delta_lw[1]) > 0.005) or (abs(delta_rw[0]) > 0.005 or abs(delta_rw[1]) > 0.005):
                motions.append("Dabbing")

    # 9. SLASHING: "Diagonal or horizontal with sweeping motion", "Shoulder and arm swing across midline", "Sweeping, wide arc", "Sudden"
    # "Forceful sweeping motion with rotation"
    left_slash_magnitude = np.linalg.norm(delta_lw[:2])
    right_slash_magnitude = np.linalg.norm(delta_rw[:2])
    if (left_slash_magnitude > 0.05 and left_slash_magnitude < 0.15) or (right_slash_magnitude > 0.05 and right_slash_magnitude < 0.15):
        if (abs(delta_lw[0]) > 0.03) or (abs(delta_rw[0]) > 0.03):
            if not any(motion in motions for motion in ["Punching"]):
                shoulder_movement = np.linalg.norm(delta_shoulder[:2])
                if shoulder_movement > 0.002:
                    motions.append("Slashing")

    # 10. PRESSING: "Downward or forward in a steady path", "Hands or arms push with tension; torso stabilizes", "Linear, controlled", "Sustained"
    # "Visible muscle engagement; movement ends in stillness or contact"
    if ((rw[1] > rh[1]+0.02 or lw[1] > lh[1]+0.02) or
        (delta_rw[1] > 0.005 or delta_lw[1] > 0.005)):
        motions.append("Pressing")

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

                # Attempt to send email notification with attachment
                ok, info = send_submission_email(user_name=user_name, user_email=user_email, slip_path=saved_path)
                if ok:
                    st.info("Notification email sent to recipient.")
                else:
                    st.warning(f"Email not sent: {info}")

    # SMTP debug panel (no sensitive info)
    with st.expander("Email settings (debug)"):
        cfg = _resolve_smtp_settings()
        masked_user = (cfg["user"] or "").strip()
        if masked_user:
            masked_user = masked_user[:2] + "***" + masked_user[-2:]
        st.write({
            "host": cfg["host"],
            "port": cfg["port"],
            "ssl": cfg["use_ssl"],
            "from": cfg["from_addr"],
            "to": cfg["to_addr"],
            "user": masked_user,
        })

uploaded_file = st.file_uploader("Upload video", type=["mp4","mov","avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

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

        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        results_data, motion_segments = [], []
        motion_summary, ongoing = {}, {}
        frame_idx, last_logged_sec = 0, -1
        prev_landmarks = None
        history = deque(maxlen=3)

        fade_text, fade_counter = [], 0
        last_overlay_update_sec = -1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            motions = []
            if results.pose_landmarks:
                motions = detect_motion_v27(results.pose_landmarks.landmark, prev_landmarks)
                prev_landmarks = results.pose_landmarks.landmark

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

                # Skeleton overlay red line + white dot
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )

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

        cap.release(); out.release(); pose.close()

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
