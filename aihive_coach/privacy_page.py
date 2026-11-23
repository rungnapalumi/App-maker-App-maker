"""Privacy Notice Page"""
import streamlit as st
from database import db


def show_privacy_notice():
    """Display privacy notice and data management"""
    
    st.title("🔒 Privacy & Data")
    
    # Privacy Notice
    st.markdown("## What We Store")
    
    st.info("""
    **AIHive Coach stores the following data locally:**
    
    - **Coaching Conversations:** Your messages and coach responses
    - **Action Plans:** Goals, insights, actions, dates, success indicators
    - **Progress Notes:** Updates you add to your action plans
    - **User ID:** Anonymous browser-based identifier (no login required)
    
    **Storage Location:** SQLite database (`aihive_coach.db`) on the server
    """)
    
    # Ephemeral Warning
    st.warning("""
    ⚠️ **Preview Environment Notice**
    
    This preview deployment uses **ephemeral storage**. Your data may be lost when:
    - The service restarts
    - The application redeploys
    - The service spins down (after 15 min inactivity on free tier)
    
    **Recommendation:** Do not enter sensitive personal information until production deployment with persistent database.
    """)
    
    # Data Deletion
    st.markdown("## Delete My Data")
    
    st.markdown("""
    You can delete all your coaching sessions and action plans from this device's anonymous ID.
    
    **This action:**
    - ✅ Deletes all your sessions and action plans
    - ✅ Cannot be undone
    - ✅ Only affects data from your current browser session
    - ❌ Does not affect data from other devices/browsers
    """)
    
    if st.button("🗑️ Delete All My Data", type="secondary"):
        st.session_state.show_delete_confirm = True
        st.rerun()
    
    # Delete Confirmation
    if st.session_state.get("show_delete_confirm", False):
        st.error("""
        ⚠️ **Confirm Data Deletion**
        
        This will permanently delete all your coaching sessions and action plans.
        This action cannot be undone.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Yes, Delete Everything", type="primary", key="confirm_delete"):
                # Delete user's data
                user_id = st.session_state.user_id
                deleted_sessions = db.delete_user_data(user_id, data_type="sessions")
                deleted_plans = db.delete_user_data(user_id, data_type="action_plans")
                
                st.session_state.show_delete_confirm = False
                st.session_state.current_session = None
                
                st.success(f"""
                ✅ **Data Deleted**
                
                - {deleted_sessions} coaching sessions removed
                - {deleted_plans} action plans removed
                
                You can start fresh by clicking "Start Coaching" in the sidebar.
                """)
                st.balloons()
        
        with col2:
            if st.button("❌ Cancel", key="cancel_delete"):
                st.session_state.show_delete_confirm = False
                st.rerun()
    
    # Rate Limiting Info
    st.markdown("---")
    st.markdown("## Rate Limiting")
    
    st.info("""
    **Message Rate Limits:**
    - Maximum 10 messages per minute per session
    - Prevents accidental spam or abuse
    - Resets automatically after 60 seconds
    
    If you hit the limit, please wait a moment before sending more messages.
    """)
    
    # Additional Info
    st.markdown("---")
    st.markdown("## Security & Privacy")
    
    st.markdown("""
    **What we DON'T collect:**
    - ❌ Your name, email, or contact information
    - ❌ Login credentials (no authentication required)
    - ❌ Tracking cookies or analytics
    - ❌ IP addresses or device fingerprints
    
    **Data Isolation:**
    - ✅ Each anonymous user ID has separate data
    - ✅ Database queries filter by user ID
    - ✅ No cross-user data access
    
    **Limitations:**
    - This is a **demo/preview environment**
    - **NOT suitable for sensitive personal data**
    - **NOT GDPR-compliant** in current state
    - For production use, authentication and persistent storage are required
    """)
    
    st.markdown("---")
    st.caption("Last updated: November 23, 2026")
