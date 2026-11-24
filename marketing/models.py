"""
Data Models for AIHive Marketing Agent
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid


class Campaign(BaseModel):
    """Represents a marketing campaign brief and its generated output"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    # Input fields
    product_service: str
    target_audience: str
    goal: str  # awareness / sign-ups / sales
    channel: str  # Facebook, LINE OA, Instagram, email, etc.
    draft_copy: Optional[str] = None
    
    # Generated output
    campaign_outline: Optional[str] = None
    content_checklist: Optional[dict] = None
    caption_variants: Optional[List[str]] = None
    metrics_to_track: Optional[List[str]] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
