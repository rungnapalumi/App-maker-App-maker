"""
Tests for Privacy Features
- Data deletion with user isolation
- Rate limiting
"""
import pytest
import time
from datetime import datetime, timezone, timedelta
from database import db
from models import CoachingSession, ActionPlan, Message


def test_delete_user_data_sessions_only():
    """Test deleting only sessions for a user"""
    # Create data for two users
    user1_id = "user_delete_test_1"
    user2_id = "user_delete_test_2"
    
    # User 1: 2 sessions
    session1 = db.create_session(user1_id, category="Test")
    session1.add_message("user", "Test message 1")
    db.save_session(session1)
    
    session2 = db.create_session(user1_id, category="Test")
    session2.add_message("user", "Test message 2")
    db.save_session(session2)
    
    # User 2: 1 session
    session3 = db.create_session(user2_id, category="Test")
    session3.add_message("user", "Test message 3")
    db.save_session(session3)
    
    # Delete user 1's sessions
    deleted = db.delete_user_data(user1_id, data_type="sessions")
    
    # Verify
    assert deleted == 2, "Should delete 2 sessions"
    
    user1_sessions = db.list_sessions(user1_id)
    assert len(user1_sessions) == 0, "User 1 should have no sessions"
    
    user2_sessions = db.list_sessions(user2_id)
    assert len(user2_sessions) == 1, "User 2 should still have 1 session"
    
    # Cleanup
    db.delete_user_data(user2_id)


def test_delete_user_data_action_plans_only():
    """Test deleting only action plans for a user"""
    # Create data
    user1_id = "user_delete_plans_1"
    user2_id = "user_delete_plans_2"
    
    # User 1: 2 plans
    session1 = db.create_session(user1_id, category="Test")
    
    plan1 = ActionPlan(
        user_id=user1_id,
        session_id=session1.id,
        goal="Goal 1",
        key_insight="Insight 1",
        next_action="Action 1",
        target_date=None,
        success_indicator="Indicator 1"
    )
    db.create_action_plan(plan1)
    
    plan2 = ActionPlan(
        user_id=user1_id,
        session_id=session1.id,
        goal="Goal 2",
        key_insight="Insight 2",
        next_action="Action 2",
        target_date=None,
        success_indicator="Indicator 2"
    )
    db.create_action_plan(plan2)
    
    # User 2: 1 plan
    session2 = db.create_session(user2_id, category="Test")
    
    plan3 = ActionPlan(
        user_id=user2_id,
        session_id=session2.id,
        goal="Goal 3",
        key_insight="Insight 3",
        next_action="Action 3",
        target_date=None,
        success_indicator="Indicator 3"
    )
    db.create_action_plan(plan3)
    
    # Delete user 1's plans
    deleted = db.delete_user_data(user1_id, data_type="action_plans")
    
    # Verify
    assert deleted == 2, "Should delete 2 action plans"
    
    user1_plans = db.list_action_plans(user1_id)
    assert len(user1_plans) == 0, "User 1 should have no plans"
    
    user2_plans = db.list_action_plans(user2_id)
    assert len(user2_plans) == 1, "User 2 should still have 1 plan"
    
    # Cleanup
    db.delete_user_data(user1_id)
    db.delete_user_data(user2_id)


def test_delete_all_user_data():
    """Test deleting all data (sessions + plans) for a user"""
    user_id = "user_delete_all"
    
    # Create session
    session = db.create_session(user_id, category="Test")
    session.add_message("user", "Test")
    db.save_session(session)
    
    # Create plan
    plan = ActionPlan(
        user_id=user_id,
        session_id=session.id,
        goal="Goal",
        key_insight="Insight",
        next_action="Action",
        target_date=None,
        success_indicator="Indicator"
    )
    db.create_action_plan(plan)
    
    # Delete all
    deleted = db.delete_user_data(user_id, data_type="all")
    
    # Verify
    assert deleted == 2, "Should delete 1 session + 1 plan"
    
    sessions = db.list_sessions(user_id)
    plans = db.list_action_plans(user_id)
    
    assert len(sessions) == 0, "User should have no sessions"
    assert len(plans) == 0, "User should have no plans"


def test_delete_user_data_isolation():
    """Test that deleting one user's data doesn't affect another user"""
    user1_id = "user_isolation_delete_1"
    user2_id = "user_isolation_delete_2"
    
    # Create data for both users
    session1 = db.create_session(user1_id, category="Test")
    session1.add_message("user", "User 1 message")
    db.save_session(session1)
    
    plan1 = ActionPlan(
        user_id=user1_id,
        session_id=session1.id,
        goal="User 1 Goal",
        key_insight="Insight",
        next_action="Action",
        target_date=None,
        success_indicator="Indicator"
    )
    db.create_action_plan(plan1)
    
    session2 = db.create_session(user2_id, category="Test")
    session2.add_message("user", "User 2 message")
    db.save_session(session2)
    
    plan2 = ActionPlan(
        user_id=user2_id,
        session_id=session2.id,
        goal="User 2 Goal",
        key_insight="Insight",
        next_action="Action",
        target_date=None,
        success_indicator="Indicator"
    )
    db.create_action_plan(plan2)
    
    # Delete user 1's data
    db.delete_user_data(user1_id)
    
    # Verify user 1 has no data
    user1_sessions = db.list_sessions(user1_id)
    user1_plans = db.list_action_plans(user1_id)
    assert len(user1_sessions) == 0, "User 1 should have no sessions after delete"
    assert len(user1_plans) == 0, "User 1 should have no plans after delete"
    
    # Verify user 2 still has data
    user2_sessions = db.list_sessions(user2_id)
    user2_plans = db.list_action_plans(user2_id)
    assert len(user2_sessions) == 1, "User 2 should still have session"
    assert len(user2_plans) == 1, "User 2 should still have plan"
    assert user2_sessions[0].user_id == user2_id, "User 2's session should be intact"
    assert user2_plans[0].user_id == user2_id, "User 2's plan should be intact"
    
    # Cleanup
    db.delete_user_data(user2_id)


def test_rate_limiting_basic():
    """Test basic rate limiting logic"""
    from coaching_page import check_rate_limit
    
    # Simulate checking rate limit
    session_id = "test_session_rate"
    
    # First message should be allowed
    allowed, count = check_rate_limit(session_id, max_messages=3, window_seconds=60)
    assert allowed == True, "First message should be allowed"
    assert count == 0, "Count should be 0 before first message"


def test_rate_limiting_with_messages():
    """Test rate limiting with message history"""
    from coaching_page import check_rate_limit
    import streamlit as st
    
    # Setup
    if "rate_limit_messages" not in st.session_state:
        st.session_state.rate_limit_messages = []
    
    session_id = "test_rate_messages"
    now = datetime.now(timezone.utc).timestamp()
    
    # Add 3 messages in last 60 seconds
    st.session_state.rate_limit_messages = [
        now - 10,  # 10 seconds ago
        now - 20,  # 20 seconds ago
        now - 30,  # 30 seconds ago
    ]
    
    # Check with limit of 3
    allowed, count = check_rate_limit(session_id, max_messages=3, window_seconds=60)
    assert allowed == False, "Should be rate limited at exactly the limit"
    assert count == 3, "Should count 3 recent messages"
    
    # Check with limit of 5 (should be allowed)
    allowed2, count2 = check_rate_limit(session_id, max_messages=5, window_seconds=60)
    assert allowed2 == True, "Should be allowed with higher limit"
    
    # Cleanup
    st.session_state.rate_limit_messages = []


def test_rate_limiting_window_expiry():
    """Test that old messages don't count toward rate limit"""
    from coaching_page import check_rate_limit
    import streamlit as st
    
    # Setup
    if "rate_limit_messages" not in st.session_state:
        st.session_state.rate_limit_messages = []
    
    session_id = "test_rate_expiry"
    now = datetime.now(timezone.utc).timestamp()
    
    # Add messages: 2 recent, 2 expired
    st.session_state.rate_limit_messages = [
        now - 10,   # Recent
        now - 20,   # Recent
        now - 70,   # Expired (70 seconds ago, window is 60)
        now - 90,   # Expired
    ]
    
    # Check with limit of 3
    allowed, count = check_rate_limit(session_id, max_messages=3, window_seconds=60)
    assert allowed == True, "Should be allowed (only 2 messages in window)"
    assert count == 2, "Should only count recent messages"
    
    # Verify expired messages were cleaned up
    assert len(st.session_state.rate_limit_messages) == 2, "Expired messages should be removed"
    
    # Cleanup
    st.session_state.rate_limit_messages = []
