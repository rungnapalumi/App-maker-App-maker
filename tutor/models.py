"""
Data Models for AIHive Tutor
Study sessions, lesson states, and progress tracking
"""
from datetime import datetime, timezone
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict
from uuid import uuid4


class LessonMessage(BaseModel):
    """A message in the tutoring conversation"""
    model_config = ConfigDict(ser_json_timedelta='iso8601')
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant"]
    content: str
    is_question: bool = False  # True if this is a check-for-understanding question
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StudySession(BaseModel):
    """A study session with topic, level, and lesson progress"""
    model_config = ConfigDict(ser_json_timedelta='iso8601')
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    topic: str
    level: Literal["Beginner", "Intermediate", "Advanced"]
    
    # Lesson state
    messages: List[LessonMessage] = Field(default_factory=list)
    current_chunk: int = 0  # Which explanation chunk (0-indexed)
    chunks_completed: int = 0  # Count of successfully explained chunks
    questions_answered_correctly: int = 0
    questions_answered_incorrectly: int = 0
    
    # Session metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    is_completed: bool = False
    
    # Summary (populated at end)
    summary_text: Optional[str] = None
    practice_questions: List[str] = Field(default_factory=list)  # 3 questions for review
    
    def add_message(self, role: str, content: str, is_question: bool = False) -> LessonMessage:
        """Add a message to the session"""
        msg = LessonMessage(role=role, content=content, is_question=is_question)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg
    
    def mark_completed(self, summary: str, practice_q: List[str]):
        """Mark session as complete with summary"""
        self.is_completed = True
        self.completed_at = datetime.now(timezone.utc)
        self.summary_text = summary
        self.practice_questions = practice_q
        self.updated_at = datetime.now(timezone.utc)
