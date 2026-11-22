"""
Coaching Workspace Page
Main chat interface for coaching conversations
"""
import streamlit as st
import asyncio
from datetime import date, timedelta
from database import db
from coach_ai import coach
from models import CoachingSession, ActionPlan


def show_coaching_workspace():
    """Main coaching workspace with chat interface"""
    
    st.title("💬 Coaching Workspace")
    
    # Initialize or load session
    if st.session_state.current_session is None:
        # Check if user wants to select a category first
        if st.session_state.current_category is None:
            st.markdown("### What would you like to work on today?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💼 Career & Leadership", use_container_width=True):
                    st.session_state.current_category = "Career & Leadership"
                    st.rerun()
                
                if st.button("🌱 Personal Growth", use_container_width=True):
                    st.session_state.current_category = "Personal Growth"
                    st.rerun()
            
            with col2:
                if st.button("🎯 Goals & Productivity", use_container_width=True):
                    st.session_state.current_category = "Goals & Productivity"
                    st.rerun()
                
                if st.button("🤔 Decision Making", use_container_width=True):
                    st.session_state.current_category = "Decision Making"
                    st.rerun()
            
            st.info("Select a category above to begin your coaching session.")
            return
        
        # Create new session with category
        session = db.create_session(
            user_id=st.session_state.user_id,
            category=st.session_state.current_category
        )
        
        # Add welcome message
        welcome = coach.get_welcome_message(session.category)
        session.add_message("assistant", welcome)
        db.save_session(session)
        
        st.session_state.current_session = session
        st.rerun()
    
    session = st.session_state.current_session
    
    # Session header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if session.category:
            st.markdown(f"**Category:** {session.category}")
    
    with col2:
        if st.button("📝 Create Action Plan"):
            st.session_state.show_action_plan_form = True
            st.rerun()
    
    with col3:
        if st.button("🔄 New Session"):
            st.session_state.current_session = None
            st.session_state.current_category = None
            st.rerun()
    
    st.markdown("---")
    
    # Display conversation
    chat_container = st.container()
    
    with chat_container:
        for message in session.messages:
            if message.role == "user":
                st.markdown(
                    f"<div class='chat-message user-message'>{message.content}</div>",
                    unsafe_allow_html=True
                )
            elif message.role == "assistant":
                st.markdown(
                    f"<div class='chat-message assistant-message'>{message.content}</div>",
                    unsafe_allow_html=True
                )
    
    # Action Plan Form (if triggered)
    if st.session_state.get("show_action_plan_form", False):
        show_action_plan_creation_form(session)
        return
    
    # Chat input
    st.markdown("---")
    
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message:",
            placeholder="Type your message here...",
            height=100,
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([4, 1])
        
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True, type="primary")
        
        if submit_button and user_input.strip():
            # Add user message and get response
            with st.spinner("Coach is thinking..."):
                # Use asyncio to run async function
                response = asyncio.run(coach.get_response(session, user_input.strip()))
                
                # Reload session from database to ensure consistency
                st.session_state.current_session = db.get_session(
                    session.id,
                    st.session_state.user_id
                )
            
            st.rerun()


def show_action_plan_creation_form(session: CoachingSession):
    """Form to create/edit action plan"""
    
    st.markdown("## 📋 Create Your Action Plan")
    
    st.info("""
    Based on our conversation, let's create your action plan. 
    You can edit any field before saving.
    """)
    
    # Generate initial plan
    if "draft_plan" not in st.session_state:
        with st.spinner("Generating your action plan..."):
            draft = coach.generate_action_plan(session)
            st.session_state.draft_plan = draft
    
    draft = st.session_state.draft_plan
    
    # Editable form
    with st.form(key="action_plan_form"):
        goal = st.text_area(
            "🎯 My Goal",
            value=draft.goal,
            height=100,
            help="Clear, specific outcome you want to achieve"
        )
        
        key_insight = st.text_area(
            "💡 My Key Insight",
            value=draft.key_insight,
            height=100,
            help="Main realization or learning from our conversation"
        )
        
        next_action = st.text_area(
            "✅ My Next Action",
            value=draft.next_action,
            height=100,
            help="Concrete first step with specific details"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_date = st.date_input(
                "📅 Target Completion Date",
                value=draft.target_date,
                min_value=date.today(),
                help="Realistic timeline"
            )
        
        with col2:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        success_indicator = st.text_area(
            "🎉 Success Indicator",
            value=draft.success_indicator,
            height=100,
            help="How will you know you succeeded?"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Action Plan", type="primary", use_container_width=True):
                # Create final action plan
                action_plan = ActionPlan(
                    user_id=session.user_id,
                    session_id=session.id,
                    goal=goal,
                    key_insight=key_insight,
                    next_action=next_action,
                    target_date=target_date,
                    success_indicator=success_indicator
                )
                
                # Save to database
                db.create_action_plan(action_plan)
                
                # Update session
                session.action_plan_id = action_plan.id
                session.is_completed = True
                db.save_session(session)
                
                # Clear state
                st.session_state.show_action_plan_form = False
                st.session_state.draft_plan = None
                
                st.success("✅ Action Plan saved successfully!")
                st.balloons()
                
                st.info("View your action plans in the 'My Action Plans' page.")
                
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state.show_action_plan_form = False
                st.session_state.draft_plan = None
                st.rerun()
