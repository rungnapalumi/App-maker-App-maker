"""
AI Coach Logic
Handles conversation with OpenAI API and demo mode fallback
"""
import random
import re
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta
from openai import OpenAI
from config import config
from models import Message, CoachingSession, ActionPlan, APIUsageLog
from prompts import COACH_SYSTEM_PROMPT, CATEGORY_PROMPTS, DEMO_MODE_RESPONSES
from database import db


class CoachAI:
    """AI Coach with OpenAI integration and demo mode"""
    
    def __init__(self):
        self.demo_mode = config.is_demo_mode()
        self.client = None
        
        if not self.demo_mode:
            try:
                self.client = OpenAI(
                    api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_API_BASE
                )
            except Exception as e:
                print(f"OpenAI initialization failed: {e}. Falling back to demo mode.")
                self.demo_mode = True
    
    def get_welcome_message(self, category: Optional[str] = None) -> str:
        """Generate welcome message"""
        base = "Hello! I'm your AIHive Coach. What would you like to work on today?"
        
        if category and category in CATEGORY_PROMPTS:
            base = f"Hello! I'm your AIHive Coach. I see you've chosen to work on **{category}**. What specifically would you like to explore in this area today?"
        
        return base
    
    async def get_response(self, session: CoachingSession, user_message: str) -> str:
        """
        Get coach response to user message
        Returns the assistant's response text
        """
        # Add user message to session
        session.add_message("user", user_message)
        
        # Check for crisis keywords
        if self._detect_crisis(user_message):
            response = self._get_crisis_response()
            session.add_message("assistant", response)
            return response
        
        # Get response based on mode
        if self.demo_mode:
            response = self._get_demo_response(session, user_message)
        else:
            response = await self._get_ai_response(session)
        
        # Add assistant response to session
        session.add_message("assistant", response)
        
        # Save session to database
        db.save_session(session)
        
        return response
    
    def _detect_crisis(self, text: str) -> bool:
        """Detect potential crisis language"""
        crisis_keywords = [
            "suicide", "suicidal", "kill myself", "end my life",
            "self-harm", "hurt myself", "want to die", "no reason to live"
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in crisis_keywords)
    
    def _get_crisis_response(self) -> str:
        """Return crisis support response"""
        return f"""I hear that you're in distress, and I want you to know that your feelings are important. However, I'm an AI coaching tool and not equipped to provide the emergency support you need right now.

**Please reach out for immediate help:**

🇹🇭 **Thailand Mental Health Hotline:** {config.THAILAND_MENTAL_HEALTH_HOTLINE} (available 24/7)

🌍 **International:**
- Emergency services: 911 (US) or your local emergency number
- Crisis Text Line (US): Text HOME to 741741
- Samaritans (UK): 116 123

**Please consider speaking with a licensed mental health professional.** They are trained to provide the support you deserve.

If you're in immediate danger, please call emergency services right away. Your life matters."""
    
    async def _get_ai_response(self, session: CoachingSession) -> str:
        """Get response from OpenAI API"""
        try:
            # Build messages for API
            messages = self._build_message_history(session)
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Log usage
            if config.LOG_API_USAGE:
                usage = response.usage
                cost = self._estimate_cost(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    config.OPENAI_MODEL
                )
                
                log = APIUsageLog(
                    session_id=session.id,
                    model=config.OPENAI_MODEL,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    estimated_cost_usd=cost
                )
                db.log_api_usage(log)
            
            return assistant_message
            
        except Exception as e:
            print(f"AI API error: {e}")
            # Fallback to demo mode on error
            return self._get_demo_response(session, session.messages[-1].content)
    
    def _get_demo_response(self, session: CoachingSession, user_message: str) -> str:
        """Generate demo mode response (heuristic)"""
        # Count user messages
        message_count = len([m for m in session.messages if m.role == "user"])
        
        # Extract key themes from user message
        message_lower = user_message.lower()
        
        # Early conversation: explore and clarify
        if message_count <= 2:
            if any(word in message_lower for word in ["want", "goal", "improve", "change"]):
                responses = [
                    "What specifically about that feels most important to you right now?",
                    "When you imagine that working out well, what does that look like?",
                    "What would be different in your life if you made progress on this?",
                ]
            elif any(word in message_lower for word in ["stuck", "struggling", "difficult", "hard"]):
                responses = [
                    "What's making this feel so challenging for you?",
                    "When did you first notice this becoming difficult?",
                    "What have you tried so far?",
                ]
            else:
                responses = [
                    "Tell me more about what brought this to mind today.",
                    "What makes this important to you right now?",
                    "Help me understand what you're hoping to explore here.",
                ]
        
        # Mid conversation: probe deeper
        elif message_count <= 5:
            if any(word in message_lower for word in ["because", "think", "feel", "believe"]):
                responses = [
                    "What evidence supports that view?",
                    "How certain are you about that? What would it take to test it?",
                    "Is that a fact, or could it be an assumption?",
                ]
            elif any(word in message_lower for word in ["can't", "won't", "never", "always"]):
                responses = [
                    "What's making you certain about that?",
                    "Has there ever been an exception to that rule?",
                    "What would need to change for a different outcome to be possible?",
                ]
            else:
                responses = [
                    "What else have you considered?",
                    "If you weren't the one facing this, what advice would you give?",
                    "What's really stopping you from taking action?",
                ]
        
        # Later conversation: move toward action
        else:
            responses = [
                "What feels like the most important next step for you?",
                "If you were to take action this week, what would that look like?",
                "What would you need in place to move forward with confidence?",
                "How will you know when you're making progress?",
            ]
        
        return random.choice(responses)
    
    def _build_message_history(self, session: CoachingSession) -> List[Dict[str, str]]:
        """Build message history for API call"""
        messages = [
            {"role": "system", "content": COACH_SYSTEM_PROMPT}
        ]
        
        # Add category context if present
        if session.category and session.category in CATEGORY_PROMPTS:
            messages.append({
                "role": "system",
                "content": CATEGORY_PROMPTS[session.category]
            })
        
        # Add conversation history
        for msg in session.messages:
            if msg.role != "system":
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return messages
    
    def generate_action_plan(self, session: CoachingSession, user_input: Optional[Dict] = None) -> ActionPlan:
        """
        Generate action plan from session
        user_input can override specific fields
        """
        if self.demo_mode or not self.client:
            return self._generate_demo_action_plan(session, user_input)
        
        try:
            # Build prompt to extract action plan
            conversation_summary = "\n".join([
                f"{m.role.upper()}: {m.content}"
                for m in session.messages[-10:]  # Last 10 messages
                if m.role != "system"
            ])
            
            prompt = f"""Based on this coaching conversation, generate a structured action plan with these exact fields:

CONVERSATION SUMMARY:
{conversation_summary}

Please extract and synthesize into this format:
GOAL: [Clear, specific outcome the user wants to achieve]
KEY_INSIGHT: [Main realization or learning from our conversation]
NEXT_ACTION: [Concrete first step with specific details]
TARGET_DATE: [Realistic date in YYYY-MM-DD format]
SUCCESS_INDICATOR: [How the user will know they succeeded]

Be specific, actionable, and use the user's own language where possible."""

            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            plan_text = response.choices[0].message.content
            parsed = self._parse_action_plan_text(plan_text)
            
            # Override with user input if provided
            if user_input:
                parsed.update(user_input)
            
            # Create ActionPlan object
            target_date_str = parsed.get("target_date", (date.today() + timedelta(days=7)).isoformat())
            try:
                target_date_obj = date.fromisoformat(target_date_str)
            except:
                target_date_obj = date.today() + timedelta(days=7)
            
            action_plan = ActionPlan(
                user_id=session.user_id,
                session_id=session.id,
                goal=parsed.get("goal", "Define my goal clearly"),
                key_insight=parsed.get("key_insight", "Reflection from our conversation"),
                next_action=parsed.get("next_action", "Take the first step"),
                target_date=target_date_obj,
                success_indicator=parsed.get("success_indicator", "Track my progress")
            )
            
            return action_plan
            
        except Exception as e:
            print(f"Action plan generation error: {e}")
            return self._generate_demo_action_plan(session, user_input)
    
    def _generate_demo_action_plan(self, session: CoachingSession, user_input: Optional[Dict] = None) -> ActionPlan:
        """Generate a demo action plan with specific, measurable fields"""
        # Extract key themes from conversation
        user_messages = [m.content for m in session.messages if m.role == "user"]
        conversation_text = " ".join(user_messages[-5:])  # Last 5 user messages
        
        # Generate specific, measurable defaults
        goal = "Define clear goal based on conversation"
        key_insight = "Identify key realization from coaching session"
        # Next Action: verb + concrete deliverable + scope
        next_action = "Schedule 30-minute session to outline first three steps"
        # Success Indicator: measurable outcome
        success_indicator = "Complete at least 2 of 3 outlined steps within one week"
        
        # Try to extract specific actions from conversation
        conversation_lower = conversation_text.lower()
        if any(word in conversation_lower for word in ["speak", "talk", "say", "communicate"]):
            goal = f"Improve communication in specific context"
            next_action = "Prepare one talking point and share it in next team interaction"
            success_indicator = "Speak up at least once in next 3 meetings or conversations"
        elif any(word in conversation_lower for word in ["habit", "routine", "practice"]):
            goal = f"Build sustainable habit or routine"
            next_action = "Set calendar reminder and complete action once tomorrow"
            success_indicator = "Complete action 5 out of 7 days this week"
        elif any(word in conversation_lower for word in ["decide", "choice", "option"]):
            goal = f"Make informed decision on pending choice"
            next_action = "List pros and cons for top 2 options by end of day"
            success_indicator = "Make final decision and communicate it within 3 days"
        elif len(conversation_text) > 20:
            # Extract topic and make it specific
            topic = conversation_text[:60].strip()
            goal = f"Take concrete action on: {topic}..."
        
        # Override with user input if provided
        if user_input:
            goal = user_input.get("goal", goal)
            key_insight = user_input.get("key_insight", key_insight)
            next_action = user_input.get("next_action", next_action)
            success_indicator = user_input.get("success_indicator", success_indicator)
        
        # NEVER invent a target date - leave as None, require user to set it
        target_date_obj = None
        if user_input and user_input.get("target_date"):
            try:
                target_date_obj = date.fromisoformat(user_input["target_date"])
            except:
                target_date_obj = None
        
        return ActionPlan(
            user_id=session.user_id,
            session_id=session.id,
            goal=goal,
            key_insight=key_insight,
            next_action=next_action,
            target_date=target_date_obj,
            success_indicator=success_indicator
        )
    
    def _parse_action_plan_text(self, text: str) -> Dict[str, str]:
        """Parse action plan from AI response text"""
        result = {}
        
        patterns = {
            "goal": r"GOAL:\s*(.+?)(?=\nKEY_INSIGHT:|\nNEXT_ACTION:|\Z)",
            "key_insight": r"KEY_INSIGHT:\s*(.+?)(?=\nNEXT_ACTION:|\nTARGET_DATE:|\Z)",
            "next_action": r"NEXT_ACTION:\s*(.+?)(?=\nTARGET_DATE:|\nSUCCESS_INDICATOR:|\Z)",
            "target_date": r"TARGET_DATE:\s*(.+?)(?=\nSUCCESS_INDICATOR:|\Z)",
            "success_indicator": r"SUCCESS_INDICATOR:\s*(.+?)(?=\Z)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        return result
    
    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate cost in USD based on token usage"""
        # Pricing as of Nov 2024 (approximate)
        pricing = {
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
            "gpt-4o": {"prompt": 0.005 / 1000, "completion": 0.015 / 1000},
            "gpt-4o-mini": {"prompt": 0.00015 / 1000, "completion": 0.0006 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
        }
        
        # Default to gpt-4o-mini pricing if model not found
        model_pricing = pricing.get(model, pricing["gpt-4o-mini"])
        
        cost = (prompt_tokens * model_pricing["prompt"] + 
                completion_tokens * model_pricing["completion"])
        
        return round(cost, 6)


# Global coach instance
coach = CoachAI()
