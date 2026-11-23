"""
AIHiveHQ Platform Homepage - Modern SaaS Landing Page
Multi-agent AI software platform
"""
import streamlit as st
from config import config


def get_api_key_status():
    """Check if OpenAI API key is configured (server-side only)"""
    # Never expose the actual key
    return "Available" if config.OPENAI_API_KEY else "Demo"


def show_homepage():
    """AIHiveHQ Platform Homepage - Modern SaaS Landing"""
    
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem 3rem 2rem;'>
        <h1 style='font-size: 3rem; font-weight: 700; margin-bottom: 1rem; background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            AIHiveHQ
        </h1>
        <h2 style='font-size: 1.5rem; color: #64748b; font-weight: 400; margin-bottom: 2rem;'>
            AI Software Platform for Personal & Professional Growth
        </h2>
        <p style='font-size: 1.1rem; color: #475569; max-width: 700px; margin: 0 auto 2rem auto; line-height: 1.6;'>
            Multiple AI agents working together to help you think clearly, act confidently, and grow daily.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Explore AI Agents", key="cta_explore", use_container_width=True, type="primary"):
            # Scroll to products (handled by anchor)
            st.markdown('<a id="products"></a>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Products Section
    st.markdown("<a id='products'></a>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🤖 AI Agent Suite</h2>", unsafe_allow_html=True)
    
    # Get API key status for Coach
    coach_status = get_api_key_status()
    
    # Product Cards Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style='background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 2rem; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;'>
                <h3 style='margin: 0; color: #1e293b;'>🧠 AIHive Coach</h3>
                <span style='background: {"#10b981" if coach_status == "Available" else "#f59e0b"}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>{coach_status}</span>
            </div>
            <p style='color: #64748b; margin-bottom: 1rem; line-height: 1.6;'>
                Personal development & goal-setting coach. GROW-inspired conversations, action plans, and progress tracking.
            </p>
            <p style='color: #94a3b8; font-size: 0.875rem; font-style: italic; margin-bottom: 1rem;'>
                {"⚠️ Demo Mode: Scripted responses (not AI-powered). Real AI available when OpenAI API key is configured." if coach_status == "Demo" else "✓ Real AI coaching powered by GPT-4o-mini."}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if coach_status == "Available" or coach_status == "Demo":
            if st.button("Launch Coach →", key="coach_launch", use_container_width=True):
                st.session_state.navigation = "AIHive Coach"
                st.rerun()
        else:
            st.button("Coming Soon", key="coach_disabled", use_container_width=True, disabled=True)
    
    with col2:
        # Get API key status for Tutor
        tutor_status = "Demo"  # Always show Demo for now
        
        st.markdown(f"""
        <div style='background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 2rem; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;'>
                <h3 style='margin: 0; color: #1e293b;'>📚 AIHive Tutor</h3>
                <span style='background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>{tutor_status}</span>
            </div>
            <p style='color: #64748b; margin-bottom: 1rem; line-height: 1.6;'>
                Study tutor that helps you understand topics step by step. Interactive lessons with check-for-understanding questions.
            </p>
            <p style='color: #94a3b8; font-size: 0.875rem; font-style: italic; margin-bottom: 1rem;'>
                {"⚠️ Demo Mode: Scripted responses (not AI-powered). Real AI available when OpenAI API key is configured." if tutor_status == "Demo" else "✓ Real AI tutoring powered by GPT-4o-mini."}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Tutor →", key="tutor_launch", use_container_width=True):
            st.session_state.navigation = "Start Learning"
            st.rerun()
    
    # Product Cards Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 2rem; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;'>
                <h3 style='margin: 0; color: #1e293b;'>💰 Finance Agent</h3>
                <span style='background: #94a3b8; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>Coming Soon</span>
            </div>
            <p style='color: #64748b; margin-bottom: 1rem; line-height: 1.6;'>
                Financial planning and analysis assistant. Budget tracking, investment guidance, and financial goal setting.
            </p>
            <p style='color: #94a3b8; font-size: 0.875rem; font-style: italic; margin-bottom: 1rem;'>
                Planned. AI-powered financial planning tools.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="finance_disabled", use_container_width=True, disabled=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; border: 2px solid #e2e8f0; border-radius: 12px; padding: 2rem; height: 100%; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            <div style='display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;'>
                <h3 style='margin: 0; color: #1e293b;'>💻 Developer Agent</h3>
                <span style='background: #94a3b8; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600;'>Coming Soon</span>
            </div>
            <p style='color: #64748b; margin-bottom: 1rem; line-height: 1.6;'>
                Code assistance and development support. Code review, debugging help, and architecture guidance.
            </p>
            <p style='color: #94a3b8; font-size: 0.875rem; font-style: italic; margin-bottom: 1rem;'>
                Planned. AI-powered development assistant.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Coming Soon", key="dev_disabled", use_container_width=True, disabled=True)
    
    # Future Vision Card
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 2rem; text-align: center; color: white;'>
        <h3 style='margin: 0 0 1rem 0; color: white;'>🌐 AI Workforce - Future Vision</h3>
        <p style='margin: 0; font-size: 1.1rem; opacity: 0.95;'>
            Coordinated multi-agent systems for complex business workflows. Coming in future releases.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Company Section
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🏢 About AIHiveHQ</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <p style='font-size: 1.1rem; color: #475569; line-height: 1.8; margin-bottom: 1.5rem;'>
                AIHiveHQ is an AI software company building practical tools for personal and professional growth.
            </p>
            <p style='font-size: 1rem; color: #64748b; line-height: 1.6;'>
                <strong>Founded by:</strong> Rungnapa Mukdasanit, Founder<br>
                <strong>Strategy:</strong> Revenue First. Scale the AI Army Later.<br>
                <strong>Mission:</strong> Help people think clearly, act confidently, and grow daily.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 1rem; color: #94a3b8;'>
        <p style='margin: 0.5rem 0;'>© 2026 AIHiveHQ. All rights reserved.</p>
        <p style='margin: 0.5rem 0; font-size: 0.875rem;'>Founded by Rungnapa Mukdasanit</p>
    </div>
    """, unsafe_allow_html=True)
