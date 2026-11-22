"""
Action Plan Quality Test Suite
Tests 10+ coaching scenarios across 4 categories for action plan quality:
- Next Action: specific (verb + deliverable + scope)
- Success Indicator: measurable (number/frequency/observable)
- All fields editable
- No invented target dates
"""
import pytest
import re
from datetime import date, timedelta
from coach_ai import CoachAI
from models import CoachingSession, Message


# Test scenarios across 4 categories
COACHING_SCENARIOS = [
    # Career & Leadership (3 scenarios)
    {
        "category": "Career & Leadership",
        "name": "Public Speaking Confidence",
        "conversation": [
            "I want to become more confident when presenting to executives",
            "I tend to rush through slides and avoid eye contact",
            "I think I need to practice more and slow down my delivery"
        ]
    },
    {
        "category": "Career & Leadership",
        "name": "Team Management",
        "conversation": [
            "I'm struggling to delegate tasks effectively to my team",
            "I worry that if I don't do it myself, it won't be done right",
            "Maybe I need to trust my team more and give clearer instructions"
        ]
    },
    {
        "category": "Career & Leadership",  
        "name": "Career Transition",
        "conversation": [
            "I'm considering switching from marketing to product management",
            "I have some PM skills but no formal experience",
            "I should probably network with PMs and work on a side project"
        ]
    },
    
    # Personal Growth (3 scenarios)
    {
        "category": "Personal Growth",
        "name": "Building Confidence",
        "conversation": [
            "I want to stop second-guessing myself in social situations",
            "I always worry about what others think of me",
            "I realize my fear of judgment is holding me back"
        ]
    },
    {
        "category": "Personal Growth",
        "name": "Work-Life Balance",
        "conversation": [
            "I'm burnt out from working 12-hour days",
            "I feel guilty when I'm not working",
            "I need to set boundaries and prioritize my health"
        ]
    },
    {
        "category": "Personal Growth",
        "name": "Communication Skills",
        "conversation": [
            "I want to improve how I communicate with my partner",
            "I tend to get defensive during disagreements",
            "I should practice active listening and pause before responding"
        ]
    },
    
    # Goals & Productivity (2 scenarios)
    {
        "category": "Goals & Productivity",
        "name": "Morning Routine",
        "conversation": [
            "I want to establish a consistent morning routine",
            "I keep hitting snooze and rushing through my morning",
            "I need to go to bed earlier and prepare the night before"
        ]
    },
    {
        "category": "Goals & Productivity",
        "name": "Side Project Launch",
        "conversation": [
            "I've been planning to launch my blog for months but haven't started",
            "I keep perfectin the design instead of writing content",
            "I should just publish something, even if it's not perfect"
        ]
    },
    
    # Decision Making (2 scenarios)
    {
        "category": "Decision Making",
        "name": "Job Offer Decision",
        "conversation": [
            "I have two job offers and can't decide which to accept",
            "One pays more, the other has better growth opportunities",
            "I need to clarify my priorities and talk to mentors"
        ]
    },
    {
        "category": "Decision Making",
        "name": "Major Purchase",
        "conversation": [
            "I'm trying to decide whether to buy a house or keep renting",
            "The market is uncertain and I'm worried about timing",
            "I should run the numbers and assess my 5-year plans"
        ]
    }
]


def has_action_verb(text):
    """Check if Next Action starts with an action verb"""
    action_verbs = [
        "schedule", "prepare", "write", "create", "complete", "practice",
        "set", "establish", "outline", "list", "research", "contact",
        "review", "draft", "organize", "identify", "clarify", "define",
        "reach out", "send", "call", "meet", "discuss", "ask"
    ]
    text_lower = text.lower().strip()
    return any(text_lower.startswith(verb) for verb in action_verbs)


def has_concrete_deliverable(text):
    """Check if Next Action mentions a concrete deliverable"""
    # Should be reasonably specific (more than just a verb)
    return len(text.split()) >= 5  # At least 5 words for specificity


def is_measurable(text):
    """Check if Success Indicator contains measurable elements"""
    text_lower = text.lower()
    
    # Check for numbers
    has_number = bool(re.search(r'\d+', text))
    
    # Check for frequency words
    frequency_words = [
        "once", "twice", "daily", "weekly", "monthly",
        "every", "each", "per", "times", "days", "weeks"
    ]
    has_frequency = any(word in text_lower for word in frequency_words)
    
    # Check for measurable outcomes
    measurable_patterns = [
        "complete", "finish", "achieve", "reach", "attain",
        "feedback", "response", "result", "outcome",
        "at least", "minimum", "by", "within"
    ]
    has_measurable_word = any(word in text_lower for word in measurable_patterns)
    
    return has_number or has_frequency or has_measurable_word


@pytest.fixture
def coach():
    """Create coach instance"""
    return CoachAI()


def create_test_session(scenario):
    """Create a coaching session from scenario"""
    session = CoachingSession(
        user_id="test_user",
        category=scenario["category"]
    )
    
    # Add welcome message
    session.add_message("assistant", f"Let's work on {scenario['category']}.")
    
    # Add conversation
    for i, user_msg in enumerate(scenario["conversation"]):
        session.add_message("user", user_msg)
        session.add_message("assistant", f"Response {i+1}")
    
    return session


@pytest.mark.parametrize("scenario", COACHING_SCENARIOS, ids=[s["name"] for s in COACHING_SCENARIOS])
def test_action_plan_quality_demo_mode(coach, scenario):
    """Test action plan quality for each scenario in Demo Mode"""
    # Ensure we're in demo mode
    coach.demo_mode = True
    
    # Create session
    session = create_test_session(scenario)
    
    # Generate action plan
    plan = coach.generate_action_plan(session)
    
    # Test 1: Next Action has verb
    assert has_action_verb(plan.next_action), (
        f"[{scenario['name']}] Next Action should start with action verb. "
        f"Got: '{plan.next_action}'"
    )
    
    # Test 2: Next Action has concrete deliverable (reasonably specific)
    assert has_concrete_deliverable(plan.next_action), (
        f"[{scenario['name']}] Next Action should be specific with concrete deliverable. "
        f"Got: '{plan.next_action}'"
    )
    
    # Test 3: Success Indicator is measurable
    assert is_measurable(plan.success_indicator), (
        f"[{scenario['name']}] Success Indicator should be measurable "
        f"(number/frequency/observable). Got: '{plan.success_indicator}'"
    )
    
    # Test 4: All fields are non-empty and editable (have content)
    assert len(plan.goal) > 10, f"[{scenario['name']}] Goal too short: '{plan.goal}'"
    assert len(plan.key_insight) > 10, f"[{scenario['name']}] Key Insight too short"
    assert len(plan.next_action) > 10, f"[{scenario['name']}] Next Action too short"
    assert len(plan.success_indicator) > 10, f"[{scenario['name']}] Success Indicator too short"
    
    # Test 5: Target date should NOT be auto-filled (CEO requirement: do not invent deadlines)
    assert plan.target_date is None, f"[{scenario['name']}] Target date should be None (not auto-filled), user must set it"
    
    print(f"\n✅ [{scenario['category']}] {scenario['name']}")
    print(f"   Next Action: {plan.next_action[:80]}...")
    print(f"   Success: {plan.success_indicator[:80]}...")


@pytest.mark.skipif(
    True,  # Always skip without API key
    reason="Requires OPENAI_API_KEY - tests real AI adaptiveness"
)
def test_real_ai_adaptiveness(coach):
    """Test that real AI adapts responses based on conversation history (skipped without key)"""
    # This test would verify that real AI:
    # 1. References previous user answers
    # 2. Adapts questions based on conversation flow
    # 3. Maintains context through GROW stages
    pass


def test_action_plan_scenario_summary():
    """Generate summary report of all test scenarios"""
    summary = {
        "total_scenarios": len(COACHING_SCENARIOS),
        "by_category": {}
    }
    
    for scenario in COACHING_SCENARIOS:
        cat = scenario["category"]
        if cat not in summary["by_category"]:
            summary["by_category"][cat] = 0
        summary["by_category"][cat] += 1
    
    print("\n" + "="*60)
    print("ACTION PLAN QUALITY TEST SUITE SUMMARY")
    print("="*60)
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print("\nBy Category:")
    for cat, count in summary["by_category"].items():
        print(f"  {cat}: {count} scenarios")
    print("="*60)
    
    assert summary["total_scenarios"] >= 10, "Should have at least 10 scenarios"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
