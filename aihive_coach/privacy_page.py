"""Privacy Notice Page"""
import streamlit as st
import json
import csv
import io
from datetime import datetime
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
    
    # Data Export
    st.markdown("## Download My Data")
    
    st.markdown("""
    Export all your coaching data for backup or personal records.
    
    **What's included:**
    - All coaching sessions and conversations
    - All action plans and progress notes
    - Only YOUR data (filtered by your anonymous user ID)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download JSON", type="primary", use_container_width=True):
            user_id = st.session_state.user_id
            export_data = db.export_user_data_json(user_id)
            
            json_str = json.dumps(export_data, indent=2)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"aihive_coach_data_{timestamp}.json"
            
            st.download_button(
                label="💾 Save JSON File",
                data=json_str,
                file_name=filename,
                mime="application/json",
                use_container_width=True
            )
            
            st.success(f"✅ JSON export ready! Click 'Save JSON File' to download.")
            st.info(f"📊 **Export Summary:**\n- {len(export_data['sessions'])} coaching sessions\n- {len(export_data['action_plans'])} action plans")
    
    with col2:
        if st.button("📥 Download CSV", type="secondary", use_container_width=True):
            user_id = st.session_state.user_id
            csv_data = db.export_user_data_csv(user_id)
            
            if csv_data["action_plans"]:
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=csv_data["action_plans"][0].keys())
                writer.writeheader()
                writer.writerows(csv_data["action_plans"])
                csv_str = output.getvalue()
                
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"aihive_coach_action_plans_{timestamp}.csv"
                
                st.download_button(
                    label="💾 Save CSV File",
                    data=csv_str,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
                
                st.success(f"✅ CSV export ready! Click 'Save CSV File' to download.")
                st.info(f"📊 **CSV contains {len(csv_data['action_plans'])} action plans**")
            else:
                st.info("No action plans to export yet. Start a coaching session to create your first plan!")
    
    st.markdown("---")
    
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
