"""
AIHive Marketing AI Logic
Demo mode only - uses rule-based templates, no fabricated data
"""
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihive_coach.config import config
from marketing.models import Campaign
from typing import Tuple, List, Dict


class MarketingAI:
    """Marketing AI that uses template-based demo mode"""
    
    def __init__(self):
        self.is_demo = True  # Always demo mode for this MVP
    
    def generate_campaign_analysis(self, campaign: Campaign) -> Tuple[str, Dict, List[str], List[str]]:
        """
        Generate campaign outline, content checklist, caption variants, and metrics.
        Returns: (campaign_outline, content_checklist_dict, caption_variants_list, metrics_list)
        
        IMPORTANT: Only computes metrics from user input, never fabricates statistics.
        """
        outline = self._generate_campaign_outline(campaign)
        checklist = self._analyze_draft_copy(campaign) if campaign.draft_copy else self._generate_empty_checklist()
        variants = self._generate_caption_variants(campaign)
        metrics = self._recommend_metrics(campaign)
        
        return (outline, checklist, variants, metrics)
    
    def _generate_campaign_outline(self, campaign: Campaign) -> str:
        """Generate a structured campaign outline based on inputs"""
        goal_map = {
            "awareness": "Build brand awareness and reach",
            "sign-ups": "Drive user registrations and sign-ups",
            "sales": "Generate direct sales and conversions"
        }
        
        goal_description = goal_map.get(campaign.goal.lower(), "Achieve marketing objectives")
        
        channel_tips = self._get_channel_tips(campaign.channel)
        
        outline = f"""**Campaign Objective:**
{goal_description} for {campaign.product_service}

**Target Audience:**
{campaign.target_audience}

**Primary Channel:**
{campaign.channel}

**Key Message Strategy:**
{self._generate_key_message(campaign)}

**Channel-Specific Plan:**
{channel_tips}

**Call to Action:**
{self._generate_cta(campaign)}

---
*Demo Note: This outline is generated from rule-based templates. Real campaign data would require market research.*
"""
        return outline
    
    def _get_channel_tips(self, channel: str) -> str:
        """Get channel-specific best practices"""
        channel_lower = channel.lower()
        
        tips = {
            "facebook": "• Use eye-catching visuals or short videos\n• Post during peak engagement times (lunch hours, evenings)\n• Encourage comments and shares\n• Consider Facebook Ads for broader reach",
            "instagram": "• Prioritize high-quality visual content\n• Use relevant hashtags (5-10 per post)\n• Leverage Stories for behind-the-scenes content\n• Engage with comments quickly",
            "line oa": "• Personalize messages to recipients\n• Use Rich Messages with images and CTAs\n• Send during appropriate hours (avoid late night)\n• Include clear value proposition",
            "email": "• Write compelling subject lines\n• Keep content scannable with headers and bullets\n• Include clear call-to-action buttons\n• Test send times for your audience",
            "twitter": "• Keep posts concise and punchy\n• Use 1-2 relevant hashtags\n• Engage in conversations\n• Tweet during active hours",
            "linkedin": "• Professional tone and industry insights\n• Share valuable content, not just promotions\n• Engage with comments professionally\n• Best times: Tuesday-Thursday, business hours"
        }
        
        for key, value in tips.items():
            if key in channel_lower:
                return value
        
        return "• Tailor content to your platform's audience\n• Post consistently\n• Monitor and respond to engagement\n• Track performance metrics"
    
    def _generate_key_message(self, campaign: Campaign) -> str:
        """Generate key message based on goal and audience"""
        goal = campaign.goal.lower()
        
        if "awareness" in goal:
            return f"Introduce {campaign.product_service} to {campaign.target_audience}, highlighting its unique value and benefits."
        elif "sign-up" in goal or "signup" in goal:
            return f"Motivate {campaign.target_audience} to register/sign up by emphasizing the value they'll receive from {campaign.product_service}."
        elif "sales" in goal or "sale" in goal:
            return f"Convert {campaign.target_audience} into customers by showcasing the benefits and offering a clear path to purchase {campaign.product_service}."
        else:
            return f"Engage {campaign.target_audience} with compelling information about {campaign.product_service}."
    
    def _generate_cta(self, campaign: Campaign) -> str:
        """Generate appropriate call-to-action"""
        goal = campaign.goal.lower()
        product = campaign.product_service
        
        cta_templates = {
            "awareness": [
                "Learn More",
                f"Discover {product}",
                "Find Out How"
            ],
            "sign-ups": [
                "Sign Up Now",
                "Join Free Today",
                "Get Started",
                "Register Now"
            ],
            "sales": [
                "Shop Now",
                "Buy Today",
                "Get Yours",
                "Order Now"
            ]
        }
        
        for key, ctas in cta_templates.items():
            if key in goal:
                return f"Recommended CTAs: {', '.join(ctas)}"
        
        return "Recommended CTAs: Learn More, Get Started, Contact Us"
    
    def _analyze_draft_copy(self, campaign: Campaign) -> Dict:
        """
        Analyze the user's draft copy.
        ONLY computes metrics from the actual text - no fabricated data.
        """
        draft = campaign.draft_copy
        
        # Compute actual metrics from the text
        word_count = len(draft.split())
        char_count = len(draft)
        line_count = len(draft.strip().split('\n'))
        
        # Check for hashtags
        hashtags = re.findall(r'#\w+', draft)
        hashtag_count = len(hashtags)
        
        # Check for emojis (basic Unicode emoji ranges)
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        emoji_count = len(emoji_pattern.findall(draft))
        
        # Check for question marks (engagement indicator)
        has_question = '?' in draft
        
        # Check for exclamation marks (excitement indicator)
        exclamation_count = draft.count('!')
        
        # Check for URLs (both with and without protocol)
        url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            r'|(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        )
        urls = url_pattern.findall(draft)
        has_link = len(urls) > 0
        
        # Analyze for CTA phrases (computed, not invented)
        cta_phrases = ['sign up', 'buy now', 'learn more', 'get started', 'shop now', 
                       'join', 'register', 'download', 'try', 'discover', 'click']
        cta_found = [phrase for phrase in cta_phrases if phrase in draft.lower()]
        has_clear_cta = len(cta_found) > 0
        
        # Channel-specific length check
        channel = campaign.channel.lower()
        length_assessment = self._assess_length(word_count, channel)
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "line_count": line_count,
            "hashtag_count": hashtag_count,
            "hashtags": hashtags,
            "emoji_count": emoji_count,
            "has_question": has_question,
            "exclamation_count": exclamation_count,
            "has_link": has_link,
            "has_clear_cta": has_clear_cta,
            "cta_phrases_found": cta_found,
            "length_assessment": length_assessment,
            "audience_fit": self._assess_audience_fit(campaign)
        }
    
    def _assess_length(self, word_count: int, channel: str) -> str:
        """Assess if word count is appropriate for the channel"""
        if "twitter" in channel:
            if word_count <= 25:
                return "✅ Good length for Twitter"
            else:
                return "⚠️ Consider shortening for Twitter (280 char limit)"
        elif "instagram" in channel:
            if word_count <= 150:
                return "✅ Good length for Instagram caption"
            else:
                return "⚠️ Instagram captions can be long, but users often don't read past 125 words"
        elif "facebook" in channel:
            if word_count <= 80:
                return "✅ Good length for Facebook (40-80 words is optimal)"
            else:
                return "ℹ️ Facebook allows longer posts, but shorter posts (40-80 words) often perform better"
        elif "email" in channel:
            if word_count <= 200:
                return "✅ Good length for email body"
            else:
                return "ℹ️ Email content is detailed - ensure it's scannable with headers and bullets"
        elif "line" in channel:
            if word_count <= 100:
                return "✅ Good length for LINE message"
            else:
                return "⚠️ Consider breaking longer content into multiple messages"
        else:
            return f"ℹ️ {word_count} words - review your platform's best practices"
    
    def _assess_audience_fit(self, campaign: Campaign) -> str:
        """Provide audience fit guidance based on inputs"""
        return f"Review if tone and language match {campaign.target_audience}'s preferences and familiarity with {campaign.product_service}"
    
    def _generate_empty_checklist(self) -> Dict:
        """Generate empty checklist when no draft provided"""
        return {
            "note": "No draft copy provided - checklist will be generated once you submit draft content"
        }
    
    def _generate_caption_variants(self, campaign: Campaign) -> List[str]:
        """
        Generate 2-3 caption variants using templates.
        These are examples based on user input, clearly marked as templates.
        """
        product = campaign.product_service
        audience = campaign.target_audience
        goal = campaign.goal.lower()
        channel = campaign.channel
        
        variants = []
        
        # Variant 1: Direct and clear
        if "awareness" in goal:
            variant1 = f"Introducing {product} for {audience}! Discover how we can help you. [Your unique benefit here] Learn more: [link]"
        elif "sign-up" in goal or "signup" in goal:
            variant1 = f"Ready to get started? Join {audience} who are already using {product}. Sign up today! [link]"
        else:  # sales
            variant1 = f"Get {product} today! Perfect for {audience}. [Highlight key benefit] Shop now: [link]"
        
        # Variant 2: Question-based engagement
        if "awareness" in goal:
            variant2 = f"Looking for [solution that {product} provides]? {product} is designed for {audience}. See what makes us different: [link]"
        elif "sign-up" in goal or "signup" in goal:
            variant2 = f"Want to [benefit from {product}]? Join now! Built for {audience}. Get started: [link]"
        else:  # sales
            variant2 = f"Why choose {product}? Because {audience} deserve the best. [Key differentiator] Order now: [link]"
        
        # Variant 3: Social proof / urgency (but no fabricated numbers)
        if "awareness" in goal:
            variant3 = f"Meet {product} - created with {audience} in mind. [Core value proposition] Discover more: [link]"
        elif "sign-up" in goal or "signup" in goal:
            variant3 = f"Join {audience} in [experiencing {product}]. Free to get started! Register here: [link]"
        else:  # sales
            variant3 = f"Limited time offer on {product}! Perfect for {audience}. [Special offer detail] Shop: [link]"
        
        variants = [variant1, variant2, variant3]
        
        # Add channel-specific guidance
        if "instagram" in channel.lower():
            variants[0] += " #YourHashtag"
        elif "twitter" in channel.lower():
            variants = [v[:180] + "... [link]" for v in variants]  # Shorter for Twitter
        
        return variants
    
    def _recommend_metrics(self, campaign: Campaign) -> List[str]:
        """Recommend metrics to track based on goal and channel"""
        goal = campaign.goal.lower()
        channel = campaign.channel.lower()
        
        # Base metrics
        base_metrics = [
            "Impressions / Reach (how many people saw the content)",
            "Engagement Rate (likes, comments, shares, clicks)",
        ]
        
        # Goal-specific metrics
        if "awareness" in goal:
            goal_metrics = [
                "Brand Mentions",
                "New Followers / Subscribers",
                "Video Views (if using video content)",
                "Post Shares / Retweets"
            ]
        elif "sign-up" in goal or "signup" in goal:
            goal_metrics = [
                "Click-Through Rate (CTR) to sign-up page",
                "Sign-up Conversion Rate",
                "Cost Per Sign-up (if using paid ads)",
                "Bounce Rate on sign-up page"
            ]
        else:  # sales
            goal_metrics = [
                "Click-Through Rate (CTR) to product page",
                "Conversion Rate (purchases / clicks)",
                "Revenue Generated",
                "Cost Per Acquisition (CPA)",
                "Average Order Value"
            ]
        
        # Channel-specific metrics
        if "email" in channel:
            channel_metrics = ["Email Open Rate", "Email Click Rate", "Unsubscribe Rate"]
        elif "facebook" in channel or "instagram" in channel:
            channel_metrics = ["Post Saves", "Profile Visits", "Link Clicks"]
        elif "line" in channel:
            channel_metrics = ["Message Delivery Rate", "Read Rate", "Response Rate"]
        elif "twitter" in channel:
            channel_metrics = ["Retweets", "Quote Tweets", "Profile Clicks"]
        else:
            channel_metrics = ["Platform-specific engagement metrics"]
        
        # Combine all metrics
        all_metrics = base_metrics + goal_metrics + channel_metrics
        
        return all_metrics


# Global marketing AI instance
marketing_ai = MarketingAI()
