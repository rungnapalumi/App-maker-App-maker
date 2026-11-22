"""
AIHive Coach MVP v0.1
Main Streamlit Application Entry Point
"""
import streamlit as st
import json
from pathlib import Path
from config import config
from database import db

# Page configuration
st.set_page_config(
    page_title="AIHiveHQ - Think Clearly. Act Confidently. Grow Daily.",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "AIHive Coach v0.1 - Think Clearly. Act Confidently. Grow Daily."
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
    st.title("🧠 AIHiveHQ")
    st.caption("Think Clearly. Act Confidently. Grow Daily.")
    
    st.markdown("")
    
    # Navigation - clean buttons
    st.markdown("### Navigation")
    
    if st.button("🏠 Home", use_container_width=True, key="nav_home"):
        st.session_state.navigation = "Home"
        st.rerun()
    
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
    
    # Mode indicator
    mode_text = config.get_model_info()
    if config.is_demo_mode():
        st.info(f"🔧 **{mode_text}**\n\nAdd `OPENAI_API_KEY` for real AI.")
    else:
        st.success(f"✅ **AI Active**\n\n{mode_text}")
    
    st.markdown("---")
    
    # Footer
    st.caption("© 2026 AIHiveHQ")
    st.caption("by Rungnapa Mukdasanit")

# Main content area
def show_home_page():
    """AIHiveHQ Company Home Page"""
    
    # Hero section
    st.markdown("<div style='text-align: center; padding: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
    st.title("Welcome to AIHiveHQ")
    st.subheader("AI Software Company. Revenue First. Scale the AI Army Later.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Company intro
    st.markdown("""
    <div class='card'>
    <p style='font-size: 1.1rem; line-height: 1.8;'>
    Founded by <strong>Rungnapa Mukdasanit</strong>, AIHiveHQ builds practical AI tools 
    that help you <strong>think clearly</strong>, <strong>act confidently</strong>, 
    and <strong>grow daily</strong>.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Product introduction
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Introducing AIHive Coach")
        st.markdown("""
        Your AI personal development coach. Clarify goals, overcome obstacles, 
        make confident decisions, and commit to meaningful action.
        
        **Key Features:**
        - Thoughtful, adaptive coaching conversations
        - Challenge assumptions and explore perspectives  
        - Create concrete Action Plans
        - Track progress over time
        """)
        
        if st.button("Try AIHive Coach →", key="cta_coach", use_container_width=True):
            st.session_state.navigation = "AIHive Coach"
            st.rerun()
    
    with col2:
        st.info("""
        **💡 What makes it different?**
        
        AIHive Coach doesn't just agree with you or offer generic advice. 
        It asks insightful questions, respectfully challenges your thinking, 
        and helps you discover your own answers.
        
        It's not a therapist—it's a thinking partner for your goals and decisions.
        """)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)


def show_coach_landing_page():
    """AIHive Coach Product Landing Page"""
    
    # Hero
    st.markdown("<div style='text-align: center; padding: 2rem 0;'>", unsafe_allow_html=True)
    st.title("🧠 AIHive Coach")
    st.subheader("Think Clearly. Act Confidently. Grow Daily.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Intro
    st.markdown("""
    <div class='card'>
    <p style='font-size: 1.1rem; text-align: center;'>
    An AI coaching experience that helps you clarify your thinking, challenge your assumptions, 
    and turn insights into action.
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
    
    # CTA
    if st.button("✨ Start Your First Session", key="cta_start", use_container_width=True):
        st.session_state.navigation = "Start Coaching"
        st.rerun()


def show_coaching_page():
    """Main Coaching Workspace - Import from pages"""
    from pages.coaching import show_coaching_workspace
    show_coaching_workspace()


def show_action_plans_page():
    """My Action Plans - Import from pages"""
    from pages.action_plans import show_action_plans
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
else:
    show_home_page()
