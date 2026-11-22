"""
Database Management for AIHive Coach
Uses SQLite for development with SQLAlchemy ORM
Implements user isolation and privacy
"""
import json
from datetime import datetime, date
from typing import List, Optional
from sqlalchemy import create_engine, Column, String, Text, Boolean, Integer, Float, DateTime, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base
from config import config
from models import CoachingSession, ActionPlan, Message, APIUsageLog

Base = declarative_base()


class DBSession(Base):
    """Database model for coaching sessions"""
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    category = Column(String, nullable=True)
    messages_json = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    action_plan_id = Column(String, nullable=True)
    is_completed = Column(Boolean, default=False)


class DBActionPlan(Base):
    """Database model for action plans"""
    __tablename__ = "action_plans"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    
    goal = Column(Text, nullable=False)
    key_insight = Column(Text, nullable=False)
    next_action = Column(Text, nullable=False)
    target_date = Column(Date, nullable=False)
    success_indicator = Column(Text, nullable=False)
    
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_completed = Column(Boolean, default=False)
    
    progress_notes_json = Column(Text, nullable=False, default="[]")
    last_reviewed_at = Column(DateTime, nullable=True)


class DBAPIUsageLog(Base):
    """Database model for API usage tracking"""
    __tablename__ = "api_usage_logs"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    model = Column(String, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    estimated_cost_usd = Column(Float, nullable=False)


class Database:
    """Database manager with privacy and isolation"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    # Session Management
    
    def create_session(self, user_id: str, category: Optional[str] = None) -> CoachingSession:
        """Create a new coaching session"""
        session = CoachingSession(user_id=user_id, category=category)
        
        db_session = DBSession(
            id=session.id,
            user_id=session.user_id,
            category=session.category,
            messages_json=json.dumps([]),
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
    
    def get_session(self, session_id: str, user_id: str) -> Optional[CoachingSession]:
        """Get a session by ID (with user isolation)"""
        db = self._get_session()
        try:
            db_session = db.query(DBSession).filter(
                DBSession.id == session_id,
                DBSession.user_id == user_id
            ).first()
            
            if not db_session:
                return None
            
            return self._db_session_to_model(db_session)
        finally:
            db.close()
    
    def save_session(self, session: CoachingSession):
        """Update an existing session"""
        db = self._get_session()
        try:
            db_session = db.query(DBSession).filter(
                DBSession.id == session.id,
                DBSession.user_id == session.user_id
            ).first()
            
            if db_session:
                db_session.messages_json = json.dumps([msg.model_dump(mode='json') for msg in session.messages])
                db_session.updated_at = session.updated_at
                db_session.action_plan_id = session.action_plan_id
                db_session.is_completed = session.is_completed
                db.commit()
        finally:
            db.close()
    
    def list_sessions(self, user_id: str, limit: int = 50) -> List[CoachingSession]:
        """List sessions for a user"""
        db = self._get_session()
        try:
            db_sessions = db.query(DBSession).filter(
                DBSession.user_id == user_id
            ).order_by(DBSession.created_at.desc()).limit(limit).all()
            
            return [self._db_session_to_model(s) for s in db_sessions]
        finally:
            db.close()
    
    # Action Plan Management
    
    def create_action_plan(self, action_plan: ActionPlan) -> ActionPlan:
        """Create a new action plan"""
        db_plan = DBActionPlan(
            id=action_plan.id,
            user_id=action_plan.user_id,
            session_id=action_plan.session_id,
            goal=action_plan.goal,
            key_insight=action_plan.key_insight,
            next_action=action_plan.next_action,
            target_date=action_plan.target_date,
            success_indicator=action_plan.success_indicator,
            created_at=action_plan.created_at,
            updated_at=action_plan.updated_at,
            is_completed=action_plan.is_completed,
            progress_notes_json=json.dumps(action_plan.progress_notes)
        )
        
        db = self._get_session()
        try:
            db.add(db_plan)
            db.commit()
            return action_plan
        finally:
            db.close()
    
    def get_action_plan(self, plan_id: str, user_id: str) -> Optional[ActionPlan]:
        """Get an action plan by ID (with user isolation)"""
        db = self._get_session()
        try:
            db_plan = db.query(DBActionPlan).filter(
                DBActionPlan.id == plan_id,
                DBActionPlan.user_id == user_id
            ).first()
            
            if not db_plan:
                return None
            
            return self._db_plan_to_model(db_plan)
        finally:
            db.close()
    
    def update_action_plan(self, action_plan: ActionPlan):
        """Update an existing action plan"""
        db = self._get_session()
        try:
            db_plan = db.query(DBActionPlan).filter(
                DBActionPlan.id == action_plan.id,
                DBActionPlan.user_id == action_plan.user_id
            ).first()
            
            if db_plan:
                db_plan.goal = action_plan.goal
                db_plan.key_insight = action_plan.key_insight
                db_plan.next_action = action_plan.next_action
                db_plan.target_date = action_plan.target_date
                db_plan.success_indicator = action_plan.success_indicator
                db_plan.updated_at = action_plan.updated_at
                db_plan.is_completed = action_plan.is_completed
                db_plan.progress_notes_json = json.dumps(action_plan.progress_notes)
                db_plan.last_reviewed_at = action_plan.last_reviewed_at
                db.commit()
        finally:
            db.close()
    
    def list_action_plans(self, user_id: str, limit: int = 50) -> List[ActionPlan]:
        """List action plans for a user"""
        db = self._get_session()
        try:
            db_plans = db.query(DBActionPlan).filter(
                DBActionPlan.user_id == user_id
            ).order_by(DBActionPlan.created_at.desc()).limit(limit).all()
            
            return [self._db_plan_to_model(p) for p in db_plans]
        finally:
            db.close()
    
    # API Usage Logging
    
    def log_api_usage(self, log: APIUsageLog):
        """Log API usage for cost tracking"""
        if not config.LOG_API_USAGE:
            return
        
        db_log = DBAPIUsageLog(
            id=log.id,
            session_id=log.session_id,
            timestamp=log.timestamp,
            model=log.model,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            estimated_cost_usd=log.estimated_cost_usd
        )
        
        db = self._get_session()
        try:
            db.add(db_log)
            db.commit()
        finally:
            db.close()
    
    def get_total_cost(self, days: int = 30) -> float:
        """Get total estimated cost for the last N days"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        db = self._get_session()
        try:
            total = db.query(DBAPIUsageLog).filter(
                DBAPIUsageLog.timestamp >= cutoff
            ).with_entities(
                db.func.sum(DBAPIUsageLog.estimated_cost_usd)
            ).scalar()
            
            return total or 0.0
        finally:
            db.close()
    
    # Helper methods
    
    def _db_session_to_model(self, db_session: DBSession) -> CoachingSession:
        """Convert DB session to Pydantic model"""
        messages_data = json.loads(db_session.messages_json)
        messages = [Message(**msg) for msg in messages_data]
        
        return CoachingSession(
            id=db_session.id,
            user_id=db_session.user_id,
            category=db_session.category,
            messages=messages,
            created_at=db_session.created_at,
            updated_at=db_session.updated_at,
            action_plan_id=db_session.action_plan_id,
            is_completed=db_session.is_completed
        )
    
    def _db_plan_to_model(self, db_plan: DBActionPlan) -> ActionPlan:
        """Convert DB action plan to Pydantic model"""
        progress_notes = json.loads(db_plan.progress_notes_json)
        
        return ActionPlan(
            id=db_plan.id,
            user_id=db_plan.user_id,
            session_id=db_plan.session_id,
            goal=db_plan.goal,
            key_insight=db_plan.key_insight,
            next_action=db_plan.next_action,
            target_date=db_plan.target_date,
            success_indicator=db_plan.success_indicator,
            created_at=db_plan.created_at,
            updated_at=db_plan.updated_at,
            is_completed=db_plan.is_completed,
            progress_notes=progress_notes,
            last_reviewed_at=db_plan.last_reviewed_at
        )


# Global database instance
db = Database()
