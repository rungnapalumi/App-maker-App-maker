"""
Tests for Coach AI Logic
Validates coaching behavior, demo mode, crisis detection, and action plan generation
"""
import pytest
from datetime import date
from coach_ai import CoachAI
from models import CoachingSession
from config import config


@pytest.fixture
def coach():
    """Create a coach instance"""
    return CoachAI()


@pytest.fixture
def test_session():
    """Create a test coaching session"""
    return CoachingSession(
        user_id="test_user",
        category="Personal Growth"
    )


def test_welcome_message(coach):
    """Test welcome message generation"""
    welcome = coach.get_welcome_message()
    
    assert "AIHive Coach" in welcome
    assert len(welcome) > 0


def test_welcome_message_with_category(coach):
    """Test welcome message with category"""
    welcome = coach.get_welcome_message(category="Career & Leadership")
    
    assert "Career & Leadership" in welcome


def test_crisis_detection(coach):
    """Test crisis keyword detection"""
    crisis_messages = [
        "I want to kill myself",
        "I'm thinking about suicide",
        "I want to end my life",
        "self-harm thoughts"
    ]
    
    for msg in crisis_messages:
        assert coach._detect_crisis(msg), f"Failed to detect crisis in: {msg}"


def test_non_crisis_messages(coach):
    """Test that normal messages don't trigger crisis detection"""
    normal_messages = [
        "I want to improve my career",
        "I'm feeling stuck with my goals",
        "I need help making a decision"
    ]
    
    for msg in normal_messages:
        assert not coach._detect_crisis(msg), f"False positive crisis detection: {msg}"


def test_crisis_response(coach):
    """Test crisis response content"""
    response = coach._get_crisis_response()
    
    assert "1323" in response  # Thailand hotline
    assert "mental health professional" in response.lower()
    assert "emergency" in response.lower()


def test_demo_mode_response(coach, test_session):
    """Test demo mode response generation"""
    coach.demo_mode = True
    
    response = coach._get_demo_response(test_session, "I want to work on my career")
    
    assert len(response) > 0
    assert isinstance(response, str)


def test_message_history_building(coach, test_session):
    """Test building message history for API"""
    test_session.add_message("user", "I want to improve my leadership skills")
    test_session.add_message("assistant", "What specifically about leadership interests you?")
    test_session.add_message("user", "I struggle with giving feedback")
    
    messages = coach._build_message_history(test_session)
    
    # Should have system prompt + category prompt + 3 conversation messages
    assert len(messages) >= 4
    assert messages[0]["role"] == "system"
    assert any(m["content"] == "I want to improve my leadership skills" for m in messages)


def test_action_plan_generation_demo_mode(coach, test_session):
    """Test action plan generation in demo mode"""
    coach.demo_mode = True
    
    test_session.add_message("user", "I want to get promoted to senior engineer")
    test_session.add_message("assistant", "What skills do you need to develop?")
    test_session.add_message("user", "I need to lead more projects and mentor juniors")
    
    plan = coach.generate_action_plan(test_session)
    
    assert plan.user_id == test_session.user_id
    assert plan.session_id == test_session.id
    assert len(plan.goal) > 0
    assert len(plan.next_action) > 0
    assert isinstance(plan.target_date, date)


def test_action_plan_with_user_overrides(coach, test_session):
    """Test action plan generation with user input overrides"""
    coach.demo_mode = True
    
    test_session.add_message("user", "I want to launch my side project")
    
    user_input = {
        "goal": "Launch MVP by end of month",
        "next_action": "Complete landing page design this week"
    }
    
    plan = coach.generate_action_plan(test_session, user_input)
    
    assert plan.goal == "Launch MVP by end of month"
    assert plan.next_action == "Complete landing page design this week"


def test_cost_estimation(coach):
    """Test API cost estimation"""
    cost = coach._estimate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        model="gpt-4o-mini"
    )
    
    # gpt-4o-mini: $0.00015/1K input, $0.0006/1K output
    expected = (1000 * 0.00015 / 1000) + (500 * 0.0006 / 1000)
    
    assert abs(cost - expected) < 0.000001


def test_parse_action_plan_text(coach):
    """Test parsing action plan from AI response"""
    text = """
GOAL: Complete the Q4 project roadmap
KEY_INSIGHT: I've been avoiding this because I fear pushback, not because I lack ideas
NEXT_ACTION: Draft the roadmap outline by Friday and schedule a review with my manager
TARGET_DATE: 2026-12-15
SUCCESS_INDICATOR: Roadmap is approved and shared with the team
"""
    
    parsed = coach._parse_action_plan_text(text)
    
    assert "Complete the Q4 project roadmap" in parsed["goal"]
    assert "avoiding this" in parsed["key_insight"]
    assert "2026-12-15" in parsed["target_date"]


# Coaching Quality Tests (for real AI mode)
@pytest.mark.skipif(config.is_demo_mode(), reason="Requires real AI API")
async def test_coaching_asks_questions_not_gives_advice(coach, test_session):
    """Test that coach asks questions rather than giving advice"""
    response = await coach.get_response(test_session, "I want to change careers but I'm scared")
    
    # Should contain question marks (asking questions)
    assert "?" in response
    
    # Should NOT contain prescriptive language
    prescriptive_words = ["you should", "you must", "you need to do"]
    assert not any(word in response.lower() for word in prescriptive_words)


@pytest.mark.skipif(config.is_demo_mode(), reason="Requires real AI API")
async def test_coaching_challenges_assumptions(coach, test_session):
    """Test that coach challenges assumptions"""
    test_session.add_message("user", "I can't start a business because I'm not an entrepreneur")
    
    response = await coach.get_response(test_session, "I'm just not the type")
    
    # Should probe the assumption
    challenge_indicators = ["what makes you", "how do you know", "what evidence", "who says"]
    assert any(indicator in response.lower() for indicator in challenge_indicators)


@pytest.mark.skipif(config.is_demo_mode(), reason="Requires real AI API")
async def test_coaching_avoids_flattery(coach, test_session):
    """Test that coach avoids excessive praise"""
    response = await coach.get_response(test_session, "I completed my first task")
    
    # Should NOT contain over-the-top praise
    flattery_words = ["amazing", "awesome", "incredible", "fantastic", "you're the best"]
    assert not any(word in response.lower() for word in flattery_words)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
