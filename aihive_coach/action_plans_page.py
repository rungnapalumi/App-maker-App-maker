"""
My Action Plans Page
View, edit, and review progress on action plans
"""
import streamlit as st
from datetime import date, datetime, timezone
from database import db
from models import ActionPlan


def show_action_plans():
    """Display user's action plans"""
    
    # Mobile-friendly styling
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
        .stButton button { font-size: 0.9rem; padding: 0.5rem; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Mobile quick navigation
    with st.expander("📱 Quick Navigation", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🏠 Home", key="mobile_nav_home_plans", use_container_width=True):
                st.session_state.navigation = "Home"
                st.rerun()
        with col2:
            if st.button("💬 Start Coaching", key="mobile_nav_coaching_plans", use_container_width=True):
                st.session_state.navigation = "Start Coaching"
                st.rerun()
        with col3:
            if st.button("💡 Coach Info", key="mobile_nav_coach_plans", use_container_width=True):
                st.session_state.navigation = "AIHive Coach"
                st.rerun()
    
    st.title("📋 My Action Plans")
    
    # Load user's action plans
    plans = db.list_action_plans(st.session_state.user_id)
    
    if not plans:
        st.info("""
        **You haven't created any action plans yet.**
        
        Start a coaching session to create your first plan!
        """)
        
        if st.button("Start Coaching Session", type="primary"):
            st.session_state.navigation = "Start Coaching"
            st.rerun()
        
        return
    
    # Display plans
    for plan in plans:
        show_action_plan_card(plan)


def show_action_plan_card(plan: ActionPlan):
    """Display a single action plan card"""
    
    # Determine status
    is_overdue = plan.target_date < date.today() and not plan.is_completed
    is_upcoming = plan.target_date >= date.today() and not plan.is_completed
    
    # Status indicator
    if plan.is_completed:
        status_badge = "✅ Completed"
        status_color = "#10b981"
    elif is_overdue:
        status_badge = "⚠️ Overdue"
        status_color = "#ef4444"
    else:
        status_badge = "🔄 In Progress"
        status_color = "#2563eb"
    
    # Card container
    with st.expander(f"**{plan.goal[:60]}{'...' if len(plan.goal) > 60 else ''}**", expanded=False):
        
        st.markdown(f"<span style='color: {status_color}; font-weight: bold;'>{status_badge}</span>", 
                   unsafe_allow_html=True)
        
        # Check if in edit mode
        is_editing = st.session_state.get(f"editing_{plan.id}", False)
        
        if is_editing:
            show_edit_form(plan)
        else:
            show_plan_view(plan)


def show_plan_view(plan: ActionPlan):
    """Display read-only view of action plan"""
    
    st.markdown("### 🎯 My Goal")
    st.markdown(plan.goal)
    
    st.markdown("### 💡 My Key Insight")
    st.markdown(plan.key_insight)
    
    st.markdown("### ✅ My Next Action")
    st.markdown(plan.next_action)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Target Date")
        st.markdown(f"**{plan.target_date.strftime('%B %d, %Y')}**")
        
        days_until = (plan.target_date - date.today()).days
        if days_until > 0:
            st.caption(f"({days_until} days from now)")
        elif days_until == 0:
            st.caption("(Today!)")
        else:
            st.caption(f"({abs(days_until)} days ago)")
    
    with col2:
        st.markdown("### 🎉 Success Indicator")
        st.markdown(plan.success_indicator)
    
    # Progress notes
    if plan.progress_notes:
        st.markdown("### 📝 Progress Notes")
        for i, note in enumerate(plan.progress_notes, 1):
            st.markdown(f"""
            <div class='progress-note'>
            <strong>Update {i}:</strong><br>
            {note}
            </div>
            """, unsafe_allow_html=True)
    
    # Metadata
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.caption(f"Created: {plan.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        if plan.last_reviewed_at:
            st.caption(f"Last Reviewed: {plan.last_reviewed_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✏️ Edit", key=f"edit_{plan.id}", use_container_width=True):
            st.session_state[f"editing_{plan.id}"] = True
            st.rerun()
    
    with col2:
        if st.button("📝 Review Progress", key=f"review_{plan.id}", use_container_width=True):
            st.session_state[f"reviewing_{plan.id}"] = True
            st.rerun()
    
    with col3:
        if not plan.is_completed:
            if st.button("✅ Mark Complete", key=f"complete_{plan.id}", use_container_width=True):
                plan.is_completed = True
                plan.updated_at = datetime.utcnow()
                db.update_action_plan(plan)
                st.success("Action plan marked as complete!")
                st.rerun()
    
    # Progress review form
    if st.session_state.get(f"reviewing_{plan.id}", False):
        show_progress_review_form(plan)


def show_edit_form(plan: ActionPlan):
    """Display edit form for action plan"""
    
    with st.form(key=f"edit_form_{plan.id}"):
        goal = st.text_area("🎯 My Goal", value=plan.goal, height=100)
        key_insight = st.text_area("💡 My Key Insight", value=plan.key_insight, height=100)
        next_action = st.text_area("✅ My Next Action", value=plan.next_action, height=100)
        target_date = st.date_input("📅 Target Date", value=plan.target_date, min_value=date.today())
        success_indicator = st.text_area("🎉 Success Indicator", value=plan.success_indicator, height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True):
                plan.goal = goal
                plan.key_insight = key_insight
                plan.next_action = next_action
                plan.target_date = target_date
                plan.success_indicator = success_indicator
                plan.updated_at = datetime.now(timezone.utc)
                
                db.update_action_plan(plan)
                st.session_state[f"editing_{plan.id}"] = False
                st.success("Changes saved!")
                st.rerun()
        
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state[f"editing_{plan.id}"] = False
                st.rerun()


def show_progress_review_form(plan: ActionPlan):
    """Display progress review form"""
    
    st.markdown("### 📊 Progress Review")
    
    st.info("""
    **Reflect on your progress:**
    - What have you accomplished since creating this plan?
    - What obstacles did you encounter?
    - What's your next step?
    """)
    
    with st.form(key=f"progress_form_{plan.id}"):
        progress_note = st.text_area(
            "What progress have you made?",
            height=150,
            placeholder="Describe your progress, obstacles, and insights..."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.form_submit_button("💾 Save Progress Note", type="primary", use_container_width=True):
                if progress_note.strip():
                    plan.add_progress_note(progress_note.strip())
                    db.update_action_plan(plan)
                    st.session_state[f"reviewing_{plan.id}"] = False
                    st.success("Progress note saved!")
                    st.rerun()
                else:
                    st.warning("Please enter a progress note.")
        
        with col2:
            if st.form_submit_button("❌ Cancel", use_container_width=True):
                st.session_state[f"reviewing_{plan.id}"] = False
                st.rerun()
