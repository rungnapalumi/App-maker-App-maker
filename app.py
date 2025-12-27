import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from moviepy.editor import VideoFileClip

st.set_page_config(page_title="Dot Motion Video Processor", layout="wide")

mp_pose = mp.solutions.pose

st.title("Dot Motion Video Processor")
st.write("Upload a video to extract pose landmarks and create a dot motion visualization")

dot_size = st.slider("Dot Size (radius in pixels)", min_value=1, max_value=5, value=2, help="Smaller dots (1-2) match the reference image better")

uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    temp_path = tfile.name
    tfile.close()
    
    if st.button("Process Video"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        cap = cv2.VideoCapture(temp_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
                # Create output video - use better codec
        output_path = "dot_motion_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )
        
        frames_processed = 0
        preview_frame = None
        
        with mp_pose.Pose(static_image_mode=False) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w, _ = frame.shape
                
                # Create black background
                output = np.zeros((h, w, 3), dtype=np.uint8)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                
                if results.pose_landmarks:
                    for lm in results.pose_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        
                        # Ensure coordinates are within bounds
                        if 0 <= cx < w and 0 <= cy < h:
                            # Draw dot (light point)
                            cv2.circle(
                                output,
                                (cx, cy),
                                radius=dot_size,
                                color=(255, 255, 255),  # white in BGR format
                                thickness=-1
                            )
                
                out.write(output)
                
                # Update preview (every 10th frame)
                if frames_processed % 10 == 0:
                    preview_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                
                frames_processed += 1
                progress = frames_processed / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frames_processed} of {total_frames}")
        
        cap.release()
        out.release()
        
        # Combine video with original audio
        status_text.text("Adding audio to video...")
        final_output_path = "dot_motion_output_with_audio.mp4"
        
        try:
            # Load original video to get audio
            original_video = VideoFileClip(temp_path)
            
            # Load processed video (without audio)
            processed_video = VideoFileClip(output_path)
            
            # Set audio from original video
            if original_video.audio is not None:
                final_video = processed_video.set_audio(original_video.audio)
            else:
                final_video = processed_video
            
            # Write final video with audio
            final_video.write_videofile(
                final_output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True
            )
            
            # Close clips
            original_video.close()
            processed_video.close()
            final_video.close()
            
            # Remove temporary video without audio
            if os.path.exists(output_path):
                os.unlink(output_path)
            
            # Update output path to final video
            output_path = final_output_path
            
        except Exception as e:
            st.warning(f"Could not add audio: {str(e)}. Video will be saved without audio.")
            # If audio extraction fails, use video without audio
        
        progress_bar.empty()
        status_text.empty()
        
        st.success("Video processing complete!")
        
        # Show preview
        if preview_frame is not None:
            st.image(preview_frame, caption="Preview of processed frame", use_container_width=True)
        
        # Provide download button
        if os.path.exists(output_path):
            with open(output_path, "rb") as video_file:
                st.download_button(
                    label="Download Processed Video",
                    data=video_file,
                    file_name="dot_motion_output.mp4",
                    mime="video/mp4"
                )
        
        # Cleanup
        os.unlink(temp_path)
        if os.path.exists("temp-audio.m4a"):
            os.unlink("temp-audio.m4a")
