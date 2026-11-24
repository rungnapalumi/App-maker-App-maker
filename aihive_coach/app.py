"""
AIHive Coach MVP v0.1
Main Streamlit Application Entry Point
"""
import streamlit as st
import json
import os
from pathlib import Path
from config import config
from database import db

# Page configuration
st.set_page_config(
    page_title="AIHive Workforce - Your Team",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="auto",  # Collapsed on mobile, expanded on desktop
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AIHive Workforce - Your Team"
    }
)

# Load custom CSS
css_path = Path(__file__).parent / "static" / "styles.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load i18n strings
i18n_path = Path(__file__).parent / "static" / "i18n" / "en.json"
if i18n_path.exists():
    with open(i18n_path) as f:
        strings = json.load(f)
else:
    strings = {}

# Initialize session state
if "user_id" not in st.session_state:
    # Generate anonymous user ID (browser-based)
    import hashlib
    import time
    st.session_state.user_id = hashlib.sha256(
        f"{time.time()}".encode()
    ).hexdigest()[:16]

if "current_session" not in st.session_state:
    st.session_state.current_session = None

if "current_category" not in st.session_state:
    st.session_state.current_category = None

# Sidebar navigation
with st.sidebar:
    st.title("🏢 AIHive")
    if config.is_demo_mode():
        st.caption("Your Team (Demo)")
    else:
        st.caption("Your Team of AI Assistants")
    
    st.markdown("")
    
    # Navigation - clean buttons
    st.markdown("### Navigation")
    
    if st.button("🏠 Workforce Hub", use_container_width=True, key="nav_home"):
        st.session_state.navigation = "Home"
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Coach")
    
    if st.button("💡 AIHive Coach", use_container_width=True, key="nav_coach"):
        st.session_state.navigation = "AIHive Coach"
        st.rerun()
    
    if st.button("💬 Start Coaching", use_container_width=True, key="nav_coaching"):
        st.session_state.navigation = "Start Coaching"
        st.rerun()
    
    if st.button("📋 My Action Plans", use_container_width=True, key="nav_plans"):
        st.session_state.navigation = "My Action Plans"
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Tutor")
    
    if st.button("📚 Start Learning", use_container_width=True, key="nav_tutor"):
        st.session_state.navigation = "Start Learning"
        st.rerun()
    
    if st.button("📖 My Study Sessions", use_container_width=True, key="nav_sessions"):
        st.session_state.navigation = "My Study Sessions"
        st.rerun()
    
    st.markdown("---")
    
    if st.button("🔒 Privacy & Data", use_container_width=True, key="nav_privacy"):
        st.session_state.navigation = "Privacy"
        st.rerun()
    
    st.markdown("---")
    
    # Mode indicator
    mode_text = config.get_model_info()
    if config.is_demo_mode():
        st.warning(f"⚠️ **Demo Mode**\n\nScripted responses (not AI-powered)")
    else:
        st.success(f"✅ **Real AI Active**\n\n{mode_text}")
    
    st.markdown("---")
    
    # Footer
    st.caption("© 2026 AIHiveHQ")
    st.caption("by Rungnapa Mukdasanit")

# Main content area
def show_home_page():
    """AIHive AI Workforce Hub"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from workforce.hub_page import show_workforce_hub
    show_workforce_hub()


def show_legacy_homepage():
    """Legacy AIHiveHQ Platform Homepage"""
    from homepage import show_homepage
    show_homepage()


def show_coach_landing_page():
    """AIHive Coach Product Landing Page"""
    
    # Mobile-specific CSS to reduce hero padding and ensure CTA is above the fold
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        h1 {
            font-size: 1.75rem !important;
            margin-bottom: 0.5rem !important;
        }
        .stSubheader {
            font-size: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero - reduced padding on mobile
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 0.5rem 0;'>
    <style>
    @media (max-width: 768px) {
        .mobile-hero { padding: 0.5rem 0 0.25rem 0 !important; }
    }
    </style>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🧠 AIHive Coach")
    st.subheader("Think Clearly. Act Confidently. Grow Daily.")
    
    # Intro
    # Prominent CTA FIRST - above fold on mobile
    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Your First Session", use_container_width=True, type="primary", key="start_first_top"):
            st.session_state.navigation = "Start Coaching"
            st.rerun()
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Intro text changes based on mode
    if config.is_demo_mode():
        intro_text = "A coaching conversation tool that helps you clarify your thinking, challenge your assumptions, and turn insights into action. <strong>(Demo Mode: Scripted responses)</strong>"
    else:
        intro_text = "An AI coaching experience that helps you clarify your thinking, challenge your assumptions, and turn insights into action."
    
    st.markdown(f"""
    <div class='card'>
    <p style='font-size: 1.1rem; text-align: center;'>
    {intro_text}
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Important disclaimer
    st.info("""
    ⚕️ **Important:** AIHive Coach is not a licensed therapist or counselor. 
    For mental health support, please consult a qualified professional.
    """)
    
    # How it works
    st.markdown("## How It Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card' style='text-align: center; height: 100%;'>
        <h3>💬 Thoughtful Conversation</h3>
        <p>One meaningful question at a time. No rigid questionnaires. 
        The coach adapts to your unique situation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center; height: 100%;'>
        <h3>🔍 Challenge Your Thinking</h3>
        <p>Distinguish facts from assumptions. Explore alternatives. 
        The coach respectfully challenges you to think deeper.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card' style='text-align: center; height: 100%;'>
        <h3>✅ Your Action Plan</h3>
        <p>Turn insights into concrete actions. Set target dates. Track progress. 
        The coach helps you commit and follow through.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Categories
    st.markdown("## What You Can Work On")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💼 Career & Leadership", use_container_width=True):
            st.session_state.current_category = "Career & Leadership"
            st.session_state.navigation = "Start Coaching"
            st.rerun()
        
        if st.button("🌱 Personal Growth", use_container_width=True):
            st.session_state.current_category = "Personal Growth"
            st.session_state.navigation = "Start Coaching"
            st.rerun()
    
    with col2:
        if st.button("🎯 Goals & Productivity", use_container_width=True):
            st.session_state.current_category = "Goals & Productivity"
            st.session_state.navigation = "Start Coaching"
            st.rerun()
        
        if st.button("🤔 Decision Making", use_container_width=True):
            st.session_state.current_category = "Decision Making"
            st.session_state.navigation = "Start Coaching"
            st.rerun()
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # CTA with mobile nav hint
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Start Coaching Now", key="cta_start", use_container_width=True, type="primary"):
            st.session_state.navigation = "Start Coaching"
            st.rerun()
    
    # Mobile navigation helper
    st.caption("📱 Mobile? Use the ☰ menu (top-left) to navigate")
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Example Coaching Journey
    st.markdown("## 📖 Example Coaching Journey")
    st.info("**Note:** This is sample data showing how a typical coaching session unfolds. Your actual coaching will be personalized to your goals.")
    
    # Journey Steps
    st.markdown("### 1️⃣ User Goal")
    st.markdown("""
    <div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2563eb;'>
    <p style='margin: 0; font-size: 1.05rem;'><em>"I want to become more confident speaking up in team meetings."</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 2️⃣ Coaching Conversation")
    st.markdown("""
    <div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;'>
    <p style='margin: 0 0 0.5rem 0;'><strong>Coach:</strong> "What specifically makes it difficult to speak up?"</p>
    <p style='margin: 0 0 0.5rem 0;'><strong>User:</strong> "I worry I'll say something wrong or that my ideas aren't good enough."</p>
    <p style='margin: 0 0 0.5rem 0;'><strong>Coach:</strong> "What evidence do you have that your ideas aren't good enough?"</p>
    <p style='margin: 0;'><strong>User:</strong> "Actually... when I have shared ideas, they've usually been well-received. I guess I'm assuming they're not good before I even try."</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 3️⃣ Key Insight")
    st.markdown("""
    <div style='background: #fef3c7; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f59e0b;'>
    <p style='margin: 0;'><strong>Insight:</strong> "I'm letting fear of judgment stop me, but my actual track record shows my contributions are valued. The barrier is my assumption, not reality."</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 4️⃣ Action Plan")
    st.markdown("""
    <div style='background: #f0fdf4; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #10b981;'>
    <p style='margin: 0 0 0.5rem 0;'><strong>Goal:</strong> Share at least one idea in every team meeting</p>
    <p style='margin: 0 0 0.5rem 0;'><strong>Next Action:</strong> Prepare one talking point before tomorrow's standup and share it in the first 10 minutes</p>
    <p style='margin: 0 0 0.5rem 0;'><strong>Target Date:</strong> Next week</p>
    <p style='margin: 0;'><strong>Success Indicator:</strong> I speak up at least once per meeting without overthinking it</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 5️⃣ Progress Review")
    st.markdown("""
    <div style='background: #f8fafc; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #7c3aed;'>
    <p style='margin: 0 0 0.5rem 0;'><strong>After One Week:</strong></p>
    <p style='margin: 0;'>"I shared ideas in 3 out of 4 meetings this week! It still feels a bit scary, but it's getting easier. My manager even built on one of my suggestions. I'm realizing my voice does add value."</p>
    </div>
    """, unsafe_allow_html=True)


def show_coaching_page():
    """Main Coaching Workspace - Import from module"""
    from coaching_page import show_coaching_workspace
    show_coaching_workspace()


def show_action_plans_page():
    """My Action Plans - Import from module"""
    from action_plans_page import show_action_plans
    show_action_plans()


# Initialize navigation if not set
if "navigation" not in st.session_state:
    st.session_state.navigation = "Home"

# Route to correct page
page = st.session_state.navigation

if page == "Home":
    show_home_page()
elif page == "AIHive Coach":
    show_coach_landing_page()
elif page == "Start Coaching":
    show_coaching_page()
elif page == "My Action Plans":
    show_action_plans_page()
elif page == "Start Learning":
    # Tutor lesson page
    try:
        from tutor.lesson_page import show_tutor_page
    except ImportError:
        # Fallback: add parent to path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from tutor.lesson_page import show_tutor_page
    show_tutor_page()
elif page == "My Study Sessions":
    # Tutor sessions page
    try:
        from tutor.sessions_page import show_sessions_page
    except ImportError:
        # Fallback: add parent to path
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        from tutor.sessions_page import show_sessions_page
    show_sessions_page()
elif page == "Privacy":
    from privacy_page import show_privacy_notice
    show_privacy_notice()
else:
    show_home_page()
