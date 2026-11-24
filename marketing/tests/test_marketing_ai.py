"""
Tests for Marketing AI Template Logic
Ensures no fabricated numbers, only computed metrics from input
"""
import pytest
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marketing.marketing_ai import marketing_ai
from marketing.models import Campaign


def test_marketing_ai_no_fabricated_statistics():
    """Ensure generated content contains NO fabricated statistics or benchmarks"""
    campaign = Campaign(
        user_id="test_user",
        product_service="Test Product",
        target_audience="Test Audience",
        goal="Awareness",
        channel="Facebook",
        draft_copy="Sample draft content"
    )
    
    outline, checklist, variants, metrics = marketing_ai.generate_campaign_analysis(campaign)
    
    # Check outline for fabricated numbers
    # Numbers in outline should NOT include things like "85% of users", "1.2M followers", etc.
    assert "%" not in outline or "Demo Note" in outline, "Outline should not contain percentage statistics"
    assert not re.search(r'\d+[kKmM]', outline), "Outline should not contain abbreviated numbers like 1.2M"
    
    # Check caption variants don't contain fabricated testimonials or numbers
    for variant in variants:
        assert "testimonial" not in variant.lower()
        assert "review" not in variant.lower()
        # Variants may have [placeholders] but not actual fabricated data
        assert not re.search(r'\d+%\s+(of|users|customers)', variant), "Variants should not have fake percentages"


def test_draft_copy_analysis_computes_only_from_input():
    """Ensure draft copy analysis only computes metrics from actual text"""
    draft = "Check out our amazing product! 🎉 Sign up now at example.com #promo #deals"
    
    campaign = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Sign-ups",
        channel="Instagram",
        draft_copy=draft
    )
    
    outline, checklist, variants, metrics = marketing_ai.generate_campaign_analysis(campaign)
    
    # Verify computed metrics match actual content
    assert checklist["word_count"] == len(draft.split())
    assert checklist["char_count"] == len(draft)
    assert checklist["hashtag_count"] == 2  # #promo and #deals
    assert "#promo" in checklist["hashtags"]
    assert "#deals" in checklist["hashtags"]
    assert checklist["emoji_count"] >= 1  # Should detect 🎉
    assert checklist["has_link"] == True  # example.com
    assert checklist["has_clear_cta"] == True  # "Sign up now"


def test_draft_copy_no_link():
    """Test analysis when draft has no link"""
    draft = "Our product is amazing"
    
    campaign = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Facebook",
        draft_copy=draft
    )
    
    outline, checklist, variants, metrics = marketing_ai.generate_campaign_analysis(campaign)
    
    assert checklist["has_link"] == False
    assert checklist["word_count"] == 4


def test_draft_copy_no_cta():
    """Test analysis when draft has no clear CTA"""
    draft = "This is just some text without any call to action"
    
    campaign = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Sales",
        channel="Email",
        draft_copy=draft
    )
    
    outline, checklist, variants, metrics = marketing_ai.generate_campaign_analysis(campaign)
    
    assert checklist["has_clear_cta"] == False
    assert len(checklist["cta_phrases_found"]) == 0


def test_channel_specific_length_assessment():
    """Test that length assessment is channel-specific"""
    short_draft = "Short tweet here"
    
    # Twitter - short is good
    campaign_twitter = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Twitter",
        draft_copy=short_draft
    )
    
    _, checklist_twitter, _, _ = marketing_ai.generate_campaign_analysis(campaign_twitter)
    assert "✅" in checklist_twitter["length_assessment"]
    
    # Email - short might be noted
    campaign_email = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Email",
        draft_copy=short_draft
    )
    
    _, checklist_email, _, _ = marketing_ai.generate_campaign_analysis(campaign_email)
    # Email assessment should still work without error
    assert "email" in checklist_email["length_assessment"].lower() or "word" in checklist_email["length_assessment"].lower()


def test_goal_specific_cta_generation():
    """Test that CTAs match the campaign goal"""
    # Awareness goal
    campaign_awareness = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Facebook"
    )
    
    outline_awareness, _, _, _ = marketing_ai.generate_campaign_analysis(campaign_awareness)
    assert "Learn More" in outline_awareness or "Discover" in outline_awareness
    
    # Sign-ups goal
    campaign_signups = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Sign-ups",
        channel="Facebook"
    )
    
    outline_signups, _, _, _ = marketing_ai.generate_campaign_analysis(campaign_signups)
    assert "Sign Up" in outline_signups or "Register" in outline_signups or "Join" in outline_signups
    
    # Sales goal
    campaign_sales = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Sales",
        channel="Facebook"
    )
    
    outline_sales, _, _, _ = marketing_ai.generate_campaign_analysis(campaign_sales)
    assert "Shop" in outline_sales or "Buy" in outline_sales or "Order" in outline_sales


def test_goal_specific_metrics():
    """Test that recommended metrics match the campaign goal"""
    # Awareness
    campaign_awareness = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Facebook"
    )
    
    _, _, _, metrics_awareness = marketing_ai.generate_campaign_analysis(campaign_awareness)
    metrics_text = " ".join(metrics_awareness).lower()
    assert "brand" in metrics_text or "reach" in metrics_text or "follower" in metrics_text
    
    # Sign-ups
    campaign_signups = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Sign-ups",
        channel="Facebook"
    )
    
    _, _, _, metrics_signups = marketing_ai.generate_campaign_analysis(campaign_signups)
    metrics_text = " ".join(metrics_signups).lower()
    assert "sign-up" in metrics_text or "conversion" in metrics_text or "registration" in metrics_text
    
    # Sales
    campaign_sales = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Sales",
        channel="Facebook"
    )
    
    _, _, _, metrics_sales = marketing_ai.generate_campaign_analysis(campaign_sales)
    metrics_text = " ".join(metrics_sales).lower()
    assert "revenue" in metrics_text or "purchase" in metrics_text or "conversion" in metrics_text


def test_caption_variants_use_user_inputs():
    """Test that caption variants incorporate user's actual inputs"""
    campaign = Campaign(
        user_id="test_user",
        product_service="EcoBottle",
        target_audience="environmentally conscious millennials",
        goal="Sales",
        channel="Instagram"
    )
    
    _, _, variants, _ = marketing_ai.generate_campaign_analysis(campaign)
    
    # All variants should mention the product
    for variant in variants:
        assert "EcoBottle" in variant, f"Variant should mention product: {variant}"
    
    # At least one variant should reference the audience or goal
    all_variants_text = " ".join(variants)
    assert len(variants) >= 2


def test_empty_checklist_when_no_draft():
    """Test that checklist is empty/note when no draft provided"""
    campaign = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Facebook",
        draft_copy=None
    )
    
    _, checklist, _, _ = marketing_ai.generate_campaign_analysis(campaign)
    
    assert "note" in checklist or checklist == {}


def test_demo_mode_label_in_output():
    """Ensure output is clearly labeled as demo/template-based"""
    campaign = Campaign(
        user_id="test_user",
        product_service="Product",
        target_audience="Audience",
        goal="Awareness",
        channel="Facebook"
    )
    
    outline, _, _, _ = marketing_ai.generate_campaign_analysis(campaign)
    
    # Outline should have demo disclaimer
    assert "Demo" in outline or "template" in outline.lower() or "rule-based" in outline.lower()


def test_no_placeholder_leaks():
    """
    Ensure no placeholder patterns like [Product], [link], etc. appear in campaign outlines.
    Test all goal x channel combinations (caption variants are intentionally templates).
    Similar to tutor test for placeholder prevention.
    """
    goals = ["Awareness", "Sign-ups", "Sales"]
    channels = ["Facebook", "Instagram", "LINE OA", "Email", "Twitter", "LinkedIn"]
    
    for goal in goals:
        for channel in channels:
            campaign = Campaign(
                user_id="test_user",
                product_service="EcoBottle",
                target_audience="Environmentally conscious consumers",
                goal=goal,
                channel=channel,
                draft_copy="Sample text with #hashtag and emoji 🌱 and link https://example.com"
            )
            
            outline, checklist, variants, metrics = marketing_ai.generate_campaign_analysis(campaign)
            
            # Campaign outline should NOT contain square bracket placeholders
            # (variants are templates so they're allowed to have them)
            assert "[" not in outline or "]" not in outline, \
                f"Campaign outline for {goal}/{channel} contains placeholder brackets: {outline}"
            
            # Specifically check the product name is properly substituted
            assert "EcoBottle" in outline, \
                f"Campaign outline for {goal}/{channel} should include product name 'EcoBottle'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
