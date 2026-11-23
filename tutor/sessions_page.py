"""
AIHive Tutor - My Study Sessions Page
View past study sessions and their summaries
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tutor.database import tutor_db
from datetime import datetime


def show_sessions_page():
    """Display user's study sessions"""
    st.title("📖 My Study Sessions")
    st.caption("Review what you've learned")
    
    user_id = st.session_state.user_id
    sessions = tutor_db.list_study_sessions(user_id, limit=50)
    
    if not sessions:
        st.info("You haven't completed any study sessions yet. Start learning to see your sessions here!")
        
        if st.button("🚀 Start First Lesson", type="primary"):
            st.session_state.navigation = "Start Learning"
            st.rerun()
        return
    
    # Filter options
    col1, col2 = st.columns([3, 1])
    with col1:
        show_filter = st.radio(
            "Show:",
            options=["All Sessions", "Completed Only", "In Progress"],
            horizontal=True,
            key="sessions_filter"
        )
    
    # Filter sessions
    if show_filter == "Completed Only":
        filtered = [s for s in sessions if s.is_completed]
    elif show_filter == "In Progress":
        filtered = [s for s in sessions if not s.is_completed]
    else:
        filtered = sessions
    
    st.markdown("---")
    
    if not filtered:
        st.info(f"No sessions match the filter '{show_filter}'")
        return
    
    # Display sessions
    for session in filtered:
        # Session card
        status_emoji = "✅" if session.is_completed else "🔄"
        status_text = "Completed" if session.is_completed else "In Progress"
        
        with st.expander(f"{status_emoji} {session.topic} ({session.level}) - {status_text}", expanded=False):
            # Metadata
            created = session.created_at.strftime("%Y-%m-%d %H:%M")
            st.caption(f"Started: {created}")
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Correct Answers", session.questions_answered_correctly)
            with col2:
                st.metric("Incorrect Answers", session.questions_answered_incorrectly)
            with col3:
                st.metric("Chunks Covered", session.chunks_completed)
            
            # Summary if completed
            if session.is_completed and session.summary_text:
                st.markdown("### 📝 Summary")
                st.info(session.summary_text)
                
                st.markdown("### 🎯 Practice Questions")
                for i, q in enumerate(session.practice_questions, 1):
                    st.markdown(f"**{i}.** {q}")
            
            # Resume button for in-progress sessions
            if not session.is_completed:
                if st.button(f"Resume Session", key=f"resume_{session.id}"):
                    st.session_state.current_tutor_session_id = session.id
                    st.session_state.navigation = "Start Learning"
                    st.rerun()
    
    st.markdown("---")
    st.caption(f"Total sessions: {len(filtered)}")
