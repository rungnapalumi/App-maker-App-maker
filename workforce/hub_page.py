"""
AIHive AI Workforce Hub
Main landing page for all AIHive agents
"""
import streamlit as st
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workforce.registry import get_all_agents, get_enabled_agents
from workforce.database import workforce_db
from aihive_coach.config import config


def show_workforce_hub():
    """Display the AI Workforce hub landing page"""
    
    # Hero section
    st.markdown("<div style='text-align: center; padding: 1rem 0;'>", unsafe_allow_html=True)
    st.title("🏢 AIHive AI Workforce")
    
    # Mode-aware subtitle
    if config.is_demo_mode():
        st.subheader("Your team of AI assistants (Demo Mode)")
    else:
        st.subheader("Your team of AI-powered assistants")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Demo mode warning
    if config.is_demo_mode():
        st.warning("""
        ⚠️ **Demo Mode - Scripted Responses**
        
        Currently running in demo mode (no `OPENAI_API_KEY` detected). 
        Available agents use scripted responses and are not AI-powered.
        
        Add `OPENAI_API_KEY` to your environment for real AI capabilities.
        """)
    
    st.markdown("---")
    
    # Agent cards
    st.markdown("## Your AI Team")
    
    agents = get_all_agents()
    
    # Display agents in a grid (2 columns)
    for i in range(0, len(agents), 2):
        col1, col2 = st.columns(2)
        
        with col1:
            if i < len(agents):
                _render_agent_card(agents[i])
        
        with col2:
            if i + 1 < len(agents):
                _render_agent_card(agents[i + 1])
    
    st.markdown("---")
    
    # Recent activity section
    st.markdown("## 📊 My Recent Activity")
    
    user_id = st.session_state.get("user_id", "")
    
    if not user_id:
        st.info("No activity yet. Start a conversation with an agent above!")
    else:
        recent_activities = workforce_db.get_recent_activity(user_id, limit=5)
        
        if not recent_activities:
            st.info("No activity yet. Start a conversation with an agent above!")
        else:
            for activity in recent_activities:
                _render_activity_item(activity)
    
    # Footer
    st.markdown("---")
    st.caption("© 2026 AIHiveHQ | Built with Streamlit + SQLite")


def _render_agent_card(agent):
    """Render a single agent card"""
    # Status badge color
    if agent.status == "Demo":
        status_color = "#10b981"  # Green
        status_emoji = "✅"
    else:
        status_color = "#f59e0b"  # Orange
        status_emoji = "⏳"
    
    # Card HTML
    card_html = f"""
    <div style='
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        height: 100%;
    '>
        <h3 style='margin: 0 0 0.5rem 0;'>{agent.name}</h3>
        <p style='color: #6b7280; margin: 0 0 1rem 0; font-size: 0.95rem;'>{agent.role}</p>
        <div style='
            display: inline-block;
            background: {status_color}22;
            color: {status_color};
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1rem;
        '>
            {status_emoji} {agent.status}
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Action button
    if agent.enabled:
        if st.button(
            f"🚀 Open {agent.name}",
            key=f"open_{agent.id}",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.navigation = agent.navigation_target
            st.rerun()
    else:
        st.button(
            f"⏳ Coming Soon",
            key=f"disabled_{agent.id}",
            use_container_width=True,
            disabled=True
        )


def _render_activity_item(activity):
    """Render a single activity item"""
    # Format timestamp
    try:
        dt = datetime.fromisoformat(activity["timestamp"].replace("Z", "+00:00"))
        time_str = dt.strftime("%b %d, %Y at %I:%M %p")
    except:
        time_str = activity["timestamp"]
    
    # Activity card
    st.markdown(f"""
    <div style='
        background: #f9fafb;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    '>
        <div style='display: flex; justify-content: space-between; align-items: start;'>
            <div>
                <strong>{activity["agent_name"]}</strong>
                <p style='margin: 0.25rem 0 0 0; color: #374151;'>{activity["title"]}</p>
                <p style='margin: 0.25rem 0 0 0; color: #9ca3af; font-size: 0.85rem;'>{time_str}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
