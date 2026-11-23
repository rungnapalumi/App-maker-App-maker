"""
Unit tests for activity isolation - ensuring users cannot see each other's data
"""
import pytest
import sqlite3
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workforce.database import WorkforceDB


@pytest.fixture
def temp_db():
    """Create a temporary database for testing"""
    # Create temp database file
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    # Initialize database with test tables
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    
    # Create coaching_sessions table
    cursor.execute("""
        CREATE TABLE coaching_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            goal TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create study_sessions table
    cursor.execute("""
        CREATE TABLE study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    
    yield path
    
    # Cleanup
    os.unlink(path)


def test_activity_isolation_basic(temp_db):
    """Test basic activity isolation between users"""
    db = WorkforceDB(db_path=temp_db)
    
    # Create test data for user A
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_a", "Goal A1", "active")
        )
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_a", "Goal A2", "active")
        )
        conn.commit()
    
    # Create test data for user B
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_b", "Goal B1", "active")
        )
        conn.commit()
    
    # User A should only see their own sessions
    user_a_activities = db.get_recent_activity("user_a", limit=10)
    assert len(user_a_activities) == 2, "User A should see 2 activities"
    assert all(activity["title"] in ["Goal A1", "Goal A2"] for activity in user_a_activities), \
        "User A should only see their own goals"
    
    # User B should only see their own sessions
    user_b_activities = db.get_recent_activity("user_b", limit=10)
    assert len(user_b_activities) == 1, "User B should see 1 activity"
    assert user_b_activities[0]["title"] == "Goal B1", "User B should only see their own goal"


def test_activity_isolation_across_agents(temp_db):
    """Test activity isolation across different agent types"""
    db = WorkforceDB(db_path=temp_db)
    
    # Create coach and tutor data for user A
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_a", "Coach Goal A", "active")
        )
        cursor.execute(
            "INSERT INTO study_sessions (user_id, topic, status) VALUES (?, ?, ?)",
            ("user_a", "Tutor Topic A", "active")
        )
        conn.commit()
    
    # Create coach and tutor data for user B
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_b", "Coach Goal B", "active")
        )
        cursor.execute(
            "INSERT INTO study_sessions (user_id, topic, status) VALUES (?, ?, ?)",
            ("user_b", "Tutor Topic B", "active")
        )
        conn.commit()
    
    # User A should see both their coach and tutor activities
    user_a_activities = db.get_recent_activity("user_a", limit=10)
    assert len(user_a_activities) == 2, "User A should see 2 activities"
    user_a_titles = [activity["title"] for activity in user_a_activities]
    assert "Coach Goal A" in user_a_titles, "User A should see their coach activity"
    assert "Tutor Topic A" in user_a_titles, "User A should see their tutor activity"
    
    # User B should see both their coach and tutor activities
    user_b_activities = db.get_recent_activity("user_b", limit=10)
    assert len(user_b_activities) == 2, "User B should see 2 activities"
    user_b_titles = [activity["title"] for activity in user_b_activities]
    assert "Coach Goal B" in user_b_titles, "User B should see their coach activity"
    assert "Tutor Topic B" in user_b_titles, "User B should see their tutor activity"
    
    # Users should NOT see each other's activities
    assert "Coach Goal B" not in user_a_titles, "User A should not see User B's coach activity"
    assert "Tutor Topic B" not in user_a_titles, "User A should not see User B's tutor activity"
    assert "Coach Goal A" not in user_b_titles, "User B should not see User A's coach activity"
    assert "Tutor Topic A" not in user_b_titles, "User B should not see User A's tutor activity"


def test_activity_by_agent_type_isolation(temp_db):
    """Test isolation when querying specific agent types"""
    db = WorkforceDB(db_path=temp_db)
    
    # Create data for two users
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        # User A data
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_a", "User A Coach", "active")
        )
        cursor.execute(
            "INSERT INTO study_sessions (user_id, topic, status) VALUES (?, ?, ?)",
            ("user_a", "User A Tutor", "active")
        )
        # User B data
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_b", "User B Coach", "active")
        )
        cursor.execute(
            "INSERT INTO study_sessions (user_id, topic, status) VALUES (?, ?, ?)",
            ("user_b", "User B Tutor", "active")
        )
        conn.commit()
    
    # Test coach activity isolation
    user_a_coach = db.get_activity_by_agent("user_a", "coach", limit=10)
    assert len(user_a_coach) == 1, "User A should see 1 coach activity"
    assert user_a_coach[0]["title"] == "User A Coach"
    
    user_b_coach = db.get_activity_by_agent("user_b", "coach", limit=10)
    assert len(user_b_coach) == 1, "User B should see 1 coach activity"
    assert user_b_coach[0]["title"] == "User B Coach"
    
    # Test tutor activity isolation
    user_a_tutor = db.get_activity_by_agent("user_a", "tutor", limit=10)
    assert len(user_a_tutor) == 1, "User A should see 1 tutor activity"
    assert user_a_tutor[0]["title"] == "User A Tutor"
    
    user_b_tutor = db.get_activity_by_agent("user_b", "tutor", limit=10)
    assert len(user_b_tutor) == 1, "User B should see 1 tutor activity"
    assert user_b_tutor[0]["title"] == "User B Tutor"


def test_empty_activity_for_new_user(temp_db):
    """Test that a new user with no activity sees empty list"""
    db = WorkforceDB(db_path=temp_db)
    
    # Create data for user A only
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
            ("user_a", "User A Goal", "active")
        )
        conn.commit()
    
    # New user C should see no activities
    user_c_activities = db.get_recent_activity("user_c", limit=10)
    assert len(user_c_activities) == 0, "New user should see no activities"


def test_activity_limit_respected(temp_db):
    """Test that activity limit is respected"""
    db = WorkforceDB(db_path=temp_db)
    
    # Create 10 coach sessions for user A
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        for i in range(10):
            cursor.execute(
                "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
                ("user_a", f"Goal {i}", "active")
            )
        conn.commit()
    
    # Request only 5 activities
    user_a_activities = db.get_recent_activity("user_a", limit=5)
    assert len(user_a_activities) == 5, "Should respect limit parameter"


def test_no_cross_user_data_leakage(temp_db):
    """Comprehensive test to ensure no cross-user data leakage"""
    db = WorkforceDB(db_path=temp_db)
    
    # Create data for 3 different users
    users = ["user_x", "user_y", "user_z"]
    
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        for user in users:
            cursor.execute(
                "INSERT INTO coaching_sessions (user_id, goal, status) VALUES (?, ?, ?)",
                (user, f"Coach {user}", "active")
            )
            cursor.execute(
                "INSERT INTO study_sessions (user_id, topic, status) VALUES (?, ?, ?)",
                (user, f"Tutor {user}", "active")
            )
        conn.commit()
    
    # Verify each user only sees their own data
    for user in users:
        activities = db.get_recent_activity(user, limit=10)
        
        # Should have exactly 2 activities (1 coach + 1 tutor)
        assert len(activities) == 2, f"{user} should have exactly 2 activities"
        
        # All activities should belong to this user
        for activity in activities:
            assert user in activity["title"], \
                f"{user} should not see other users' activities: {activity['title']}"
        
        # Should not see other users' data
        for other_user in users:
            if other_user != user:
                for activity in activities:
                    assert other_user not in activity["title"], \
                        f"{user} should not see {other_user}'s data"
