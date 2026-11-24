"""
AIHive Marketing Agent - My Campaigns Page
View past campaigns and their analyses
"""
import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marketing.database import marketing_db
from datetime import datetime


def show_campaigns_list_page():
    """Display user's campaign list"""
    st.title("📋 My Campaigns")
    st.caption("Review your past campaign analyses")
    
    user_id = st.session_state.user_id
    campaigns = marketing_db.list_campaigns(user_id, limit=50)
    
    if not campaigns:
        st.info("You haven't created any campaigns yet. Start by creating your first campaign!")
        
        if st.button("🚀 Create First Campaign", type="primary"):
            st.session_state.navigation = "Marketing Agent"
            st.session_state.show_campaign_result = None
            st.rerun()
        return
    
    st.markdown(f"**Total Campaigns:** {len(campaigns)}")
    st.markdown("---")
    
    # Display campaigns
    for campaign in campaigns:
        with st.expander(
            f"📢 {campaign.product_service} → {campaign.target_audience}", 
            expanded=False
        ):
            # Metadata
            created = campaign.created_at.strftime("%Y-%m-%d %H:%M")
            st.caption(f"Created: {created}")
            
            # Campaign details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Goal:** {campaign.goal}")
                st.markdown(f"**Channel:** {campaign.channel}")
            with col2:
                draft_status = "✅ Included" if campaign.draft_copy else "❌ Not provided"
                st.markdown(f"**Draft Copy:** {draft_status}")
            
            st.markdown("---")
            
            # Campaign Outline
            st.markdown("#### 🎯 Campaign Outline")
            with st.container():
                st.markdown(campaign.campaign_outline)
            
            # Content Checklist (if draft was provided)
            if campaign.draft_copy and campaign.content_checklist:
                st.markdown("#### ✅ Content Analysis")
                checklist = campaign.content_checklist
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Words", checklist["word_count"])
                with col2:
                    st.metric("Hashtags", checklist["hashtag_count"])
                with col3:
                    st.metric("Emojis", checklist["emoji_count"])
                with col4:
                    cta_status = "✅" if checklist["has_clear_cta"] else "⚠️"
                    st.metric("CTA", cta_status)
            
            # Caption Variants
            st.markdown("#### 💬 Caption Variants")
            for i, variant in enumerate(campaign.caption_variants, 1):
                st.text_area(
                    f"Variant {i}",
                    value=variant,
                    height=80,
                    key=f"campaign_{campaign.id}_variant_{i}",
                    label_visibility="collapsed"
                )
            
            # Metrics
            st.markdown("#### 📈 Recommended Metrics")
            metrics_text = "\n".join([f"• {m}" for m in campaign.metrics_to_track])
            st.markdown(metrics_text)
    
    st.markdown("---")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Create New Campaign", type="primary", use_container_width=True):
            st.session_state.navigation = "Marketing Agent"
            st.session_state.show_campaign_result = None
            st.rerun()
    with col2:
        if st.button("🏠 Back to Workforce Hub", use_container_width=True):
            st.session_state.navigation = "Workforce Hub"
            st.rerun()
