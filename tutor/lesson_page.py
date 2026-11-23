"""
AIHive Tutor - Main Lesson Page
Interactive study session interface
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihive_coach.config import config
from tutor.database import tutor_db
from tutor.tutor_ai import tutor_ai
from tutor.models import StudySession


def show_topic_picker():
    """Show the topic and level selection interface"""
    st.title("📚 AIHive Tutor")
    st.subheader("Learn step-by-step with an AI study tutor")
    
    # Mode indicator
    if config.is_demo_mode():
        st.warning("⚠️ **Demo Mode** - Scripted responses (not AI-powered). Add `OPENAI_API_KEY` for real AI tutoring.")
    else:
        st.success("✅ **Real AI Active** - Powered by GPT-4o-mini")
    
    st.markdown("---")
    
    # Topic selection
    st.markdown("### What do you want to learn today?")
    
    # Preset topics (for demo convenience)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🌱 Photosynthesis", use_container_width=True):
            st.session_state.tutor_topic_input = "Photosynthesis"
    
    with col2:
        if st.button("🐍 Python Programming", use_container_width=True):
            st.session_state.tutor_topic_input = "Python Programming"
    
    with col3:
        if st.button("🧮 Algebra", use_container_width=True):
            st.session_state.tutor_topic_input = "Algebra"
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Custom topic input
    topic = st.text_input(
        "Or type your own topic:",
        key="tutor_topic_input",
        placeholder="e.g., World War 2, Spanish verbs, Calculus, Climate change..."
    )
    
    # Level selection
    st.markdown("### Your level:")
    level = st.radio(
        "Choose your current knowledge level:",
        options=["Beginner", "Intermediate", "Advanced"],
        horizontal=True,
        key="tutor_level"
    )
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Learning", type="primary", use_container_width=True):
            if not topic or len(topic.strip()) < 3:
                st.error("Please enter a topic (at least 3 characters)")
            else:
                # Create new study session
                user_id = st.session_state.user_id
                session = tutor_db.create_study_session(user_id, topic.strip(), level)
                
                # Generate first lesson chunk
                first_message = tutor_ai.start_lesson(session)
                session.add_message("assistant", first_message, is_question=True)
                tutor_db.save_study_session(session)
                
                st.session_state.current_tutor_session_id = session.id
                st.rerun()


def show_lesson_interface():
    """Show the active lesson conversation"""
    session_id = st.session_state.get("current_tutor_session_id")
    
    if not session_id:
        show_topic_picker()
        return
    
    # Load session
    user_id = st.session_state.user_id
    session = tutor_db.get_study_session(session_id, user_id)
    
    if not session:
        st.error("Session not found")
        if st.button("Back to Topic Picker"):
            st.session_state.current_tutor_session_id = None
            st.rerun()
        return
    
    # Header
    st.title(f"📚 Learning: {session.topic}")
    st.caption(f"Level: {session.level} | Questions correct: {session.questions_answered_correctly}")
    
    # Mode indicator (compact)
    mode_text = "Demo Mode" if config.is_demo_mode() else "Real AI"
    st.caption(f"Mode: {mode_text}")
    
    # End lesson button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col3:
        if st.button("✅ End Lesson", key="end_lesson_btn"):
            # Generate summary
            summary, questions = tutor_ai.generate_summary(session)
            session.mark_completed(summary, questions)
            tutor_db.save_study_session(session)
            st.session_state.show_tutor_summary = True
            st.rerun()
    
    with col2:
        if st.button("🏠 New Topic", key="new_topic_btn"):
            st.session_state.current_tutor_session_id = None
            st.rerun()
    
    st.markdown("---")
    
    # Show summary if completed
    if session.is_completed and st.session_state.get("show_tutor_summary", False):
        show_session_summary(session)
        return
    
    # Display conversation
    for msg in session.messages:
        if msg.role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)
        else:
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg.content)
    
    # Input area
    user_input = st.chat_input("Type your answer here...", key="tutor_chat_input")
    
    if user_input:
        # Add user message
        session.add_message("user", user_input)
        
        # Check if last assistant message was a question
        if session.messages[-2].is_question:
            # Process as answer to question
            is_correct, feedback = tutor_ai.process_answer(session, user_input)
            
            if is_correct:
                session.questions_answered_correctly += 1
            else:
                session.questions_answered_incorrectly += 1
            
            # Give feedback
            session.add_message("assistant", feedback)
            
            # Continue lesson
            next_chunk = tutor_ai.continue_lesson(session, is_correct)
            
            # Check if lesson should end
            if "completed this lesson" in next_chunk.lower() or "wrap" in next_chunk.lower():
                # Generate summary
                summary, questions = tutor_ai.generate_summary(session)
                session.mark_completed(summary, questions)
                st.session_state.show_tutor_summary = True
            else:
                session.add_message("assistant", next_chunk, is_question=True)
                session.chunks_completed += 1
        
        else:
            # General response (shouldn't happen often)
            session.add_message("assistant", "I see. Let's continue...")
        
        tutor_db.save_study_session(session)
        st.rerun()


def show_session_summary(session: StudySession):
    """Display end-of-session summary"""
    st.success("🎉 Lesson Complete!")
    
    st.markdown("## 📝 Session Summary")
    
    st.markdown(f"""
    <div style='background: #f0fdf4; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #10b981;'>
    <h3>What you learned:</h3>
    <p>{session.summary_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Questions Correct", session.questions_answered_correctly)
    with col2:
        st.metric("Questions Incorrect", session.questions_answered_incorrectly)
    with col3:
        st.metric("Concepts Covered", session.chunks_completed)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Practice questions
    st.markdown("## 🎯 Practice Questions for Review")
    st.info("Review these questions later to reinforce what you learned!")
    
    for i, q in enumerate(session.practice_questions, 1):
        st.markdown(f"**{i}.** {q}")
    
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📚 Start New Topic", use_container_width=True, type="primary"):
            st.session_state.current_tutor_session_id = None
            st.session_state.show_tutor_summary = False
            st.rerun()
    
    with col2:
        if st.button("📖 View My Study Sessions", use_container_width=True):
            st.session_state.navigation = "My Study Sessions"
            st.session_state.show_tutor_summary = False
            st.rerun()


def show_tutor_page():
    """Main entry point for tutor page"""
    if st.session_state.get("current_tutor_session_id"):
        show_lesson_interface()
    else:
        show_topic_picker()
