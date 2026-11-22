"""
Tests for Database Layer
Validates user isolation, CRUD operations, and data integrity
"""
import pytest
import os
import tempfile
from datetime import date, timedelta
from database import Database
from models import CoachingSession, ActionPlan, Message


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    db = Database(db_path=path)
    yield db
    
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


def test_create_session(test_db):
    """Test creating a new coaching session"""
    session = test_db.create_session(user_id="user123", category="Career & Leadership")
    
    assert session.id is not None
    assert session.user_id == "user123"
    assert session.category == "Career & Leadership"
    assert len(session.messages) == 0


def test_get_session(test_db):
    """Test retrieving a session"""
    # Create session
    session = test_db.create_session(user_id="user123", category="Personal Growth")
    session_id = session.id
    
    # Retrieve it
    retrieved = test_db.get_session(session_id, "user123")
    
    assert retrieved is not None
    assert retrieved.id == session_id
    assert retrieved.user_id == "user123"
    assert retrieved.category == "Personal Growth"


def test_user_isolation_sessions(test_db):
    """Test that users cannot access each other's sessions"""
    # User1 creates a session
    session1 = test_db.create_session(user_id="user1", category="Goals & Productivity")
    
    # User2 tries to access it
    retrieved = test_db.get_session(session1.id, "user2")
    
    assert retrieved is None


def test_save_session_with_messages(test_db):
    """Test saving session with messages"""
    session = test_db.create_session(user_id="user123")
    
    session.add_message("user", "I want to work on my career goals.")
    session.add_message("assistant", "What specifically about your career would you like to explore?")
    
    test_db.save_session(session)
    
    # Retrieve and verify
    retrieved = test_db.get_session(session.id, "user123")
    
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].content == "I want to work on my career goals."
    assert retrieved.messages[1].role == "assistant"


def test_list_sessions(test_db):
    """Test listing sessions for a user"""
    # Create multiple sessions
    test_db.create_session(user_id="user1", category="Career & Leadership")
    test_db.create_session(user_id="user1", category="Personal Growth")
    test_db.create_session(user_id="user2", category="Goals & Productivity")
    
    # List user1's sessions
    sessions = test_db.list_sessions("user1")
    
    assert len(sessions) == 2
    assert all(s.user_id == "user1" for s in sessions)


def test_create_action_plan(test_db):
    """Test creating an action plan"""
    session = test_db.create_session(user_id="user123")
    
    action_plan = ActionPlan(
        user_id="user123",
        session_id=session.id,
        goal="Complete project by Friday",
        key_insight="I need to prioritize and delegate",
        next_action="Schedule team meeting tomorrow at 10am",
        target_date=date.today() + timedelta(days=5),
        success_indicator="Project delivered and team is aligned"
    )
    
    created = test_db.create_action_plan(action_plan)
    
    assert created.id is not None
    assert created.goal == "Complete project by Friday"


def test_get_action_plan(test_db):
    """Test retrieving an action plan"""
    session = test_db.create_session(user_id="user123")
    
    plan = ActionPlan(
        user_id="user123",
        session_id=session.id,
        goal="Test goal",
        key_insight="Test insight",
        next_action="Test action",
        target_date=date.today(),
        success_indicator="Test success"
    )
    
    created = test_db.create_action_plan(plan)
    
    # Retrieve it
    retrieved = test_db.get_action_plan(created.id, "user123")
    
    assert retrieved is not None
    assert retrieved.goal == "Test goal"


def test_user_isolation_action_plans(test_db):
    """Test that users cannot access each other's action plans"""
    session = test_db.create_session(user_id="user1")
    
    plan = ActionPlan(
        user_id="user1",
        session_id=session.id,
        goal="Private goal",
        key_insight="Private insight",
        next_action="Private action",
        target_date=date.today(),
        success_indicator="Private success"
    )
    
    created = test_db.create_action_plan(plan)
    
    # User2 tries to access it
    retrieved = test_db.get_action_plan(created.id, "user2")
    
    assert retrieved is None


def test_update_action_plan(test_db):
    """Test updating an action plan"""
    session = test_db.create_session(user_id="user123")
    
    plan = ActionPlan(
        user_id="user123",
        session_id=session.id,
        goal="Original goal",
        key_insight="Original insight",
        next_action="Original action",
        target_date=date.today(),
        success_indicator="Original success"
    )
    
    created = test_db.create_action_plan(plan)
    
    # Update it
    created.goal = "Updated goal"
    created.is_completed = True
    test_db.update_action_plan(created)
    
    # Retrieve and verify
    retrieved = test_db.get_action_plan(created.id, "user123")
    
    assert retrieved.goal == "Updated goal"
    assert retrieved.is_completed is True


def test_action_plan_progress_notes(test_db):
    """Test adding progress notes to action plan"""
    session = test_db.create_session(user_id="user123")
    
    plan = ActionPlan(
        user_id="user123",
        session_id=session.id,
        goal="Goal with progress",
        key_insight="Insight",
        next_action="Action",
        target_date=date.today(),
        success_indicator="Success"
    )
    
    created = test_db.create_action_plan(plan)
    
    # Add progress notes
    created.add_progress_note("Made good progress today")
    created.add_progress_note("Hit a roadblock, need to adjust approach")
    test_db.update_action_plan(created)
    
    # Retrieve and verify
    retrieved = test_db.get_action_plan(created.id, "user123")
    
    assert len(retrieved.progress_notes) == 2
    assert "good progress" in retrieved.progress_notes[0]
    assert retrieved.last_reviewed_at is not None


def test_list_action_plans(test_db):
    """Test listing action plans for a user"""
    session1 = test_db.create_session(user_id="user1")
    session2 = test_db.create_session(user_id="user2")
    
    # Create plans for different users
    plan1 = ActionPlan(
        user_id="user1",
        session_id=session1.id,
        goal="User 1 goal",
        key_insight="Insight 1",
        next_action="Action 1",
        target_date=date.today(),
        success_indicator="Success 1"
    )
    
    plan2 = ActionPlan(
        user_id="user2",
        session_id=session2.id,
        goal="User 2 goal",
        key_insight="Insight 2",
        next_action="Action 2",
        target_date=date.today(),
        success_indicator="Success 2"
    )
    
    test_db.create_action_plan(plan1)
    test_db.create_action_plan(plan2)
    
    # List plans for user1
    plans = test_db.list_action_plans("user1")
    
    assert len(plans) == 1
    assert plans[0].goal == "User 1 goal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
