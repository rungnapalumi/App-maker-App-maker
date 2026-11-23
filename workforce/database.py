"""
Workforce Activity Tracking Database
Tracks recent user activity across all agents
"""
import sqlite3
from typing import List, Dict
from pathlib import Path


class WorkforceDB:
    """Database for workforce hub activity tracking"""
    
    def __init__(self, db_path: str = "aihive_coach.db"):
        """Initialize database connection"""
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        # Note: We're reusing the existing aihive_coach.db database
        # Coach sessions are already in 'coaching_sessions' table
        # Tutor sessions are already in 'study_sessions' table
        # We just need to query them, no new tables needed
        pass
    
    def get_recent_activity(self, user_id: str, limit: int = 5) -> List[Dict]:
        """
        Get recent activity for a user across all agents
        Returns list of activity items sorted by timestamp descending
        """
        activities = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get recent coach sessions
            try:
                cursor.execute("""
                    SELECT 
                        'coach' as agent_type,
                        id,
                        goal,
                        created_at as timestamp,
                        status
                    FROM coaching_sessions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                for row in cursor.fetchall():
                    activities.append({
                        "agent_type": "coach",
                        "agent_name": "AIHive Coach",
                        "id": row["id"],
                        "title": row["goal"] or "New coaching session",
                        "timestamp": row["timestamp"],
                        "status": row["status"]
                    })
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                pass
            
            # Get recent tutor sessions
            try:
                cursor.execute("""
                    SELECT 
                        'tutor' as agent_type,
                        id,
                        topic,
                        created_at as timestamp,
                        status
                    FROM study_sessions
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit))
                
                for row in cursor.fetchall():
                    activities.append({
                        "agent_type": "tutor",
                        "agent_name": "AIHive Tutor",
                        "id": row["id"],
                        "title": row["topic"] or "New study session",
                        "timestamp": row["timestamp"],
                        "status": row["status"]
                    })
            except sqlite3.OperationalError:
                # Table doesn't exist yet
                pass
        
        # Sort all activities by timestamp descending
        activities.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Return only the requested limit
        return activities[:limit]
    
    def get_activity_by_agent(self, user_id: str, agent_type: str, limit: int = 5) -> List[Dict]:
        """
        Get recent activity for a specific agent type
        
        Args:
            user_id: User identifier
            agent_type: 'coach' or 'tutor'
            limit: Maximum number of items to return
        """
        activities = []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if agent_type == "coach":
                try:
                    cursor.execute("""
                        SELECT 
                            id,
                            goal,
                            created_at as timestamp,
                            status
                        FROM coaching_sessions
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (user_id, limit))
                    
                    for row in cursor.fetchall():
                        activities.append({
                            "id": row["id"],
                            "title": row["goal"] or "New coaching session",
                            "timestamp": row["timestamp"],
                            "status": row["status"]
                        })
                except sqlite3.OperationalError:
                    pass
            
            elif agent_type == "tutor":
                try:
                    cursor.execute("""
                        SELECT 
                            id,
                            topic,
                            created_at as timestamp,
                            status
                        FROM study_sessions
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (user_id, limit))
                    
                    for row in cursor.fetchall():
                        activities.append({
                            "id": row["id"],
                            "title": row["topic"] or "New study session",
                            "timestamp": row["timestamp"],
                            "status": row["status"]
                        })
                except sqlite3.OperationalError:
                    pass
        
        return activities


# Global instance
workforce_db = WorkforceDB()
