"""
Database Management for AIHive Tutor
Extends the shared SQLite database with tutor tables
"""
import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, String, Text, Boolean, Integer, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihive_coach.config import config
from tutor.models import StudySession, LessonMessage

Base = declarative_base()


class DBStudySession(Base):
    """Database model for study sessions"""
    __tablename__ = "study_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    topic = Column(String, nullable=False)
    level = Column(String, nullable=False)
    
    messages_json = Column(Text, nullable=False)
    current_chunk = Column(Integer, default=0)
    chunks_completed = Column(Integer, default=0)
    questions_answered_correctly = Column(Integer, default=0)
    questions_answered_incorrectly = Column(Integer, default=0)
    
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    is_completed = Column(Boolean, default=False)
    
    summary_text = Column(Text, nullable=True)
    practice_questions_json = Column(Text, nullable=False, default="[]")


class TutorDatabase:
    """Database manager for tutor data with privacy and isolation"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def create_study_session(self, user_id: str, topic: str, level: str) -> StudySession:
        """Create a new study session"""
        session = StudySession(user_id=user_id, topic=topic, level=level)
        
        db_session = DBStudySession(
            id=session.id,
            user_id=session.user_id,
            topic=session.topic,
            level=session.level,
            messages_json=json.dumps([]),
            current_chunk=0,
            chunks_completed=0,
            questions_answered_correctly=0,
            questions_answered_incorrectly=0,
            created_at=session.created_at,
            updated_at=session.updated_at,
            is_completed=False
        )
        
        db = self._get_session()
        try:
            db.add(db_session)
            db.commit()
            return session
        finally:
            db.close()
    
    def get_study_session(self, session_id: str, user_id: str) -> Optional[StudySession]:
        """Get a study session by ID (with user isolation)"""
        db = self._get_session()
        try:
            db_session = db.query(DBStudySession).filter(
                DBStudySession.id == session_id,
                DBStudySession.user_id == user_id
            ).first()
            
            if not db_session:
                return None
            
            return self._db_session_to_model(db_session)
        finally:
            db.close()
    
    def save_study_session(self, session: StudySession):
        """Update an existing study session"""
        db = self._get_session()
        try:
            db_session = db.query(DBStudySession).filter(
                DBStudySession.id == session.id,
                DBStudySession.user_id == session.user_id
            ).first()
            
            if db_session:
                db_session.messages_json = json.dumps([msg.model_dump(mode='json') for msg in session.messages])
                db_session.current_chunk = session.current_chunk
                db_session.chunks_completed = session.chunks_completed
                db_session.questions_answered_correctly = session.questions_answered_correctly
                db_session.questions_answered_incorrectly = session.questions_answered_incorrectly
                db_session.updated_at = session.updated_at
                db_session.completed_at = session.completed_at
                db_session.is_completed = session.is_completed
                db_session.summary_text = session.summary_text
                db_session.practice_questions_json = json.dumps(session.practice_questions)
                db.commit()
        finally:
            db.close()
    
    def list_study_sessions(self, user_id: str, limit: int = 50) -> List[StudySession]:
        """List study sessions for a user"""
        db = self._get_session()
        try:
            db_sessions = db.query(DBStudySession).filter(
                DBStudySession.user_id == user_id
            ).order_by(DBStudySession.created_at.desc()).limit(limit).all()
            
            return [self._db_session_to_model(s) for s in db_sessions]
        finally:
            db.close()
    
    def delete_user_data(self, user_id: str) -> int:
        """Delete all tutor data for a user - returns count deleted"""
        db = self._get_session()
        try:
            sessions = db.query(DBStudySession).filter(DBStudySession.user_id == user_id).all()
            deleted = len(sessions)
            for s in sessions:
                db.delete(s)
            db.commit()
            return deleted
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def _db_session_to_model(self, db_session: DBStudySession) -> StudySession:
        """Convert DB session to Pydantic model"""
        messages_data = json.loads(db_session.messages_json)
        messages = [LessonMessage(**msg) for msg in messages_data]
        
        practice_questions = json.loads(db_session.practice_questions_json) if db_session.practice_questions_json else []
        
        return StudySession(
            id=db_session.id,
            user_id=db_session.user_id,
            topic=db_session.topic,
            level=db_session.level,
            messages=messages,
            current_chunk=db_session.current_chunk,
            chunks_completed=db_session.chunks_completed,
            questions_answered_correctly=db_session.questions_answered_correctly,
            questions_answered_incorrectly=db_session.questions_answered_incorrectly,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
            completed_at=db_session.completed_at,
            is_completed=db_session.is_completed,
            summary_text=db_session.summary_text,
            practice_questions=practice_questions
        )


# Global database instance
tutor_db = TutorDatabase()
