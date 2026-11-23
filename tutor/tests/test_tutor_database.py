"""
Unit tests for TutorDatabase - data isolation and CRUD operations
"""
import pytest
import os
import tempfile
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tutor.database import TutorDatabase
from tutor.models import StudySession


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = TutorDatabase(db_path=path)
    yield db
    os.unlink(path)


@pytest.fixture
def sample_session(temp_db):
    """Create a sample study session"""
    return temp_db.create_study_session(
        user_id="user123",
        topic="Photosynthesis",
        level="Beginner"
    )


def test_create_study_session(temp_db):
    """Test creating a new study session"""
    session = temp_db.create_study_session(
        user_id="user1",
        topic="Python Programming",
        level="Intermediate"
    )
    
    assert session.id is not None
    assert session.user_id == "user1"
    assert session.topic == "Python Programming"
    assert session.level == "Intermediate"
    assert session.messages == []
    assert session.current_chunk == 0
    assert session.chunks_completed == 0
    assert not session.is_completed


def test_get_study_session(temp_db, sample_session):
    """Test retrieving a study session"""
    retrieved = temp_db.get_study_session(sample_session.id, sample_session.user_id)
    
    assert retrieved is not None
    assert retrieved.id == sample_session.id
    assert retrieved.topic == sample_session.topic
    assert retrieved.level == sample_session.level


def test_get_nonexistent_session(temp_db):
    """Test retrieving a non-existent session returns None"""
    result = temp_db.get_study_session("fake_id", "user1")
    assert result is None


def test_save_study_session(temp_db, sample_session):
    """Test updating a study session"""
    # Modify session
    sample_session.add_message("assistant", "Welcome to the lesson!")
    sample_session.chunks_completed = 1
    sample_session.questions_answered_correctly = 1
    
    # Save
    temp_db.save_study_session(sample_session)
    
    # Retrieve and verify
    retrieved = temp_db.get_study_session(sample_session.id, sample_session.user_id)
    assert len(retrieved.messages) == 1
    assert retrieved.chunks_completed == 1
    assert retrieved.questions_answered_correctly == 1


def test_list_study_sessions(temp_db):
    """Test listing study sessions for a user"""
    # Create multiple sessions
    temp_db.create_study_session("user1", "Topic A", "Beginner")
    temp_db.create_study_session("user1", "Topic B", "Intermediate")
    temp_db.create_study_session("user2", "Topic C", "Advanced")
    
    # List user1's sessions
    sessions = temp_db.list_study_sessions("user1")
    assert len(sessions) == 2
    
    # List user2's sessions
    sessions = temp_db.list_study_sessions("user2")
    assert len(sessions) == 1


def test_data_isolation(temp_db):
    """Test that users cannot access each other's data"""
    # Create session for user1
    session1 = temp_db.create_study_session("user1", "Private Topic", "Beginner")
    
    # Try to access as user2
    retrieved = temp_db.get_study_session(session1.id, "user2")
    assert retrieved is None
    
    # Verify user1 can still access it
    retrieved = temp_db.get_study_session(session1.id, "user1")
    assert retrieved is not None


def test_session_completion(temp_db, sample_session):
    """Test marking a session as completed"""
    summary = "You learned about photosynthesis!"
    questions = [
        "What is photosynthesis?",
        "Why do plants need sunlight?",
        "What gas do plants release?"
    ]
    
    sample_session.mark_completed(summary, questions)
    temp_db.save_study_session(sample_session)
    
    # Retrieve and verify
    retrieved = temp_db.get_study_session(sample_session.id, sample_session.user_id)
    assert retrieved.is_completed
    assert retrieved.summary_text == summary
    assert len(retrieved.practice_questions) == 3
    assert retrieved.completed_at is not None


def test_delete_user_data(temp_db):
    """Test deleting all data for a user"""
    # Create sessions for two users
    temp_db.create_study_session("user1", "Topic A", "Beginner")
    temp_db.create_study_session("user1", "Topic B", "Intermediate")
    temp_db.create_study_session("user2", "Topic C", "Advanced")
    
    # Delete user1's data
    deleted_count = temp_db.delete_user_data("user1")
    assert deleted_count == 2
    
    # Verify user1's data is gone
    sessions = temp_db.list_study_sessions("user1")
    assert len(sessions) == 0
    
    # Verify user2's data is intact
    sessions = temp_db.list_study_sessions("user2")
    assert len(sessions) == 1


def test_message_persistence(temp_db, sample_session):
    """Test that messages are properly persisted"""
    # Add multiple messages
    sample_session.add_message("assistant", "Welcome!", is_question=False)
    sample_session.add_message("user", "Hello!")
    sample_session.add_message("assistant", "What is 2+2?", is_question=True)
    sample_session.add_message("user", "4")
    
    temp_db.save_study_session(sample_session)
    
    # Retrieve and verify
    retrieved = temp_db.get_study_session(sample_session.id, sample_session.user_id)
    assert len(retrieved.messages) == 4
    assert retrieved.messages[0].role == "assistant"
    assert retrieved.messages[0].content == "Welcome!"
    assert retrieved.messages[2].is_question is True


def test_progress_tracking(temp_db, sample_session):
    """Test progress tracking fields"""
    sample_session.chunks_completed = 3
    sample_session.questions_answered_correctly = 5
    sample_session.questions_answered_incorrectly = 2
    
    temp_db.save_study_session(sample_session)
    
    retrieved = temp_db.get_study_session(sample_session.id, sample_session.user_id)
    assert retrieved.chunks_completed == 3
    assert retrieved.questions_answered_correctly == 5
    assert retrieved.questions_answered_incorrectly == 2


def test_sessions_ordered_by_creation(temp_db):
    """Test that sessions are returned newest first"""
    import time
    
    # Create sessions with slight delay
    s1 = temp_db.create_study_session("user1", "First", "Beginner")
    time.sleep(0.01)
    s2 = temp_db.create_study_session("user1", "Second", "Intermediate")
    time.sleep(0.01)
    s3 = temp_db.create_study_session("user1", "Third", "Advanced")
    
    sessions = temp_db.list_study_sessions("user1")
    
    # Newest first
    assert sessions[0].topic == "Third"
    assert sessions[1].topic == "Second"
    assert sessions[2].topic == "First"
