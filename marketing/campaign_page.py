"""
AIHive Marketing Agent - Campaign Creation Page
Create and analyze marketing campaigns
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marketing.database import marketing_db
from marketing.marketing_ai import marketing_ai
from marketing.models import Campaign
from datetime import datetime


def show_campaign_page():
    """Display campaign creation and analysis page"""
    st.title("📢 Marketing Agent")
    st.caption("Campaign strategy and content optimization (Demo)")
    
    # Demo mode warning
    st.info("""
    **Demo Mode (Scripted Templates)**
    
    This is a rule-based demo. Analysis is generated from templates, not AI.
    No data-driven benchmarks, statistics, or partnerships are used.
    """)
    
    st.markdown("---")
    
    # Campaign Brief Form
    st.markdown("## 📝 Campaign Brief")
    
    with st.form("campaign_brief"):
        product_service = st.text_input(
            "Product / Service",
            placeholder="e.g., Online Language Learning App",
            help="What are you marketing?"
        )
        
        target_audience = st.text_input(
            "Target Audience",
            placeholder="e.g., Working professionals aged 25-40 in Bangkok",
            help="Who are you trying to reach?"
        )
        
        goal = st.selectbox(
            "Campaign Goal",
            options=["Awareness", "Sign-ups", "Sales"],
            help="What do you want to achieve?"
        )
        
        channel = st.selectbox(
            "Primary Channel",
            options=["Facebook", "Instagram", "LINE OA", "Email", "Twitter", "LinkedIn", "Other"],
            help="Where will you run this campaign?"
        )
        
        draft_copy = st.text_area(
            "Draft Post / Ad Copy (Optional)",
            placeholder="Paste your draft content here for analysis...",
            help="If you have draft content, we'll analyze it",
            height=150
        )
        
        submitted = st.form_submit_button("🚀 Generate Campaign Analysis", type="primary", use_container_width=True)
    
    if submitted:
        # Validation
        if not product_service or not target_audience:
            st.error("⚠️ Please fill in Product/Service and Target Audience")
            return
        
        # Create campaign
        user_id = st.session_state.user_id
        campaign = Campaign(
            user_id=user_id,
            product_service=product_service,
            target_audience=target_audience,
            goal=goal,
            channel=channel,
            draft_copy=draft_copy if draft_copy else None
        )
        
        # Generate analysis
        with st.spinner("Generating campaign analysis..."):
            outline, checklist, variants, metrics = marketing_ai.generate_campaign_analysis(campaign)
            
            campaign.campaign_outline = outline
            campaign.content_checklist = checklist
            campaign.caption_variants = variants
            campaign.metrics_to_track = metrics
        
        # Save to database
        marketing_db.create_campaign(campaign)
        
        st.success("✅ Campaign analysis generated!")
        st.session_state.show_campaign_result = campaign.id
    
    # Show result if just generated
    if st.session_state.get("show_campaign_result"):
        campaign_id = st.session_state.show_campaign_result
        user_id = st.session_state.user_id
        campaign = marketing_db.get_campaign(campaign_id, user_id)
        
        if campaign:
            _display_campaign_result(campaign)


def _display_campaign_result(campaign: Campaign):
    """Display the campaign analysis result"""
    st.markdown("---")
    st.markdown("## 📊 Campaign Analysis")
    
    # Campaign Outline
    st.markdown("### 🎯 Campaign Outline")
    st.markdown(campaign.campaign_outline)
    
    # Content Checklist
    st.markdown("### ✅ Content Checklist")
    if campaign.draft_copy:
        checklist = campaign.content_checklist
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", checklist["word_count"])
        with col2:
            st.metric("Characters", checklist["char_count"])
        with col3:
            st.metric("Hashtags", checklist["hashtag_count"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Emojis", checklist["emoji_count"])
        with col2:
            cta_status = "✅ Yes" if checklist["has_clear_cta"] else "⚠️ No"
            st.metric("Has CTA", cta_status)
        with col3:
            link_status = "✅ Yes" if checklist["has_link"] else "ℹ️ No"
            st.metric("Has Link", link_status)
        
        st.markdown("**Length Assessment:**")
        st.info(checklist["length_assessment"])
        
        if checklist["has_clear_cta"]:
            st.markdown("**CTA Phrases Found:**")
            st.success(", ".join(checklist["cta_phrases_found"]))
        else:
            st.warning("⚠️ No clear call-to-action detected. Consider adding: 'Sign Up', 'Learn More', 'Shop Now', etc.")
        
        if checklist["hashtags"]:
            st.markdown("**Hashtags Used:**")
            st.write(" ".join(checklist["hashtags"]))
        
        st.markdown("**Audience Fit:**")
        st.info(checklist["audience_fit"])
    else:
        st.info("No draft copy provided. Submit your draft content to get a detailed checklist analysis.")
    
    # Caption Variants
    st.markdown("### 💬 Caption Variants (Template Examples)")
    st.caption("These are template-based examples. Customize them with your brand voice and specific details.")
    
    for i, variant in enumerate(campaign.caption_variants, 1):
        with st.expander(f"Variant {i}", expanded=(i == 1)):
            st.text_area(
                f"Caption {i}",
                value=variant,
                height=100,
                key=f"variant_{i}_{campaign.id}",
                label_visibility="collapsed"
            )
    
    # Metrics to Track
    st.markdown("### 📈 Metrics to Track")
    st.caption("Recommended metrics based on your goal and channel")
    
    for metric in campaign.metrics_to_track:
        st.markdown(f"• {metric}")
    
    # Action buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 View All Campaigns", use_container_width=True):
            st.session_state.navigation = "My Campaigns"
            st.rerun()
    with col2:
        if st.button("➕ Create New Campaign", use_container_width=True):
            st.session_state.show_campaign_result = None
            st.rerun()
