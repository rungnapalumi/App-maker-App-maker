"""
Data Models for AIHive Coach
Uses Pydantic for validation and serialization
"""
from datetime import datetime, date, timezone
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class Message(BaseModel):
    """A single message in a coaching conversation"""
    model_config = ConfigDict(ser_json_timedelta='iso8601')
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoachingSession(BaseModel):
    """A coaching conversation session"""
    model_config = ConfigDict(ser_json_timedelta='iso8601')
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    category: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action_plan_id: Optional[str] = None
    is_completed: bool = False
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a message to the session"""
        msg = Message(role=role, content=content)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg


class ActionPlan(BaseModel):
    """User's action plan generated from coaching session"""
    model_config = ConfigDict(ser_json_timedelta='iso8601')
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    session_id: str
    
    # Core action plan fields
    goal: str
    key_insight: str
    next_action: str
    target_date: Optional[date] = None  # User must set this
    success_indicator: str
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_completed: bool = False
    
    # Progress reviews
    progress_notes: List[str] = Field(default_factory=list)
    last_reviewed_at: Optional[datetime] = None
    
    def add_progress_note(self, note: str):
        """Add a progress review note"""
        self.progress_notes.append(note)
        self.last_reviewed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)


class APIUsageLog(BaseModel):
    """Log entry for AI API usage and cost tracking"""
    model_config = ConfigDict(ser_json_timedelta='iso8601')
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
