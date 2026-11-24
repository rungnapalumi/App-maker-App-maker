"""
Database Management for AIHive Marketing Agent
Extends the shared SQLite database with marketing tables
"""
import json
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aihive_coach.config import config
from marketing.models import Campaign

Base = declarative_base()


class DBCampaign(Base):
    """Database model for marketing campaigns"""
    __tablename__ = "marketing_campaigns"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    
    # Input fields
    product_service = Column(String, nullable=False)
    target_audience = Column(String, nullable=False)
    goal = Column(String, nullable=False)
    channel = Column(String, nullable=False)
    draft_copy = Column(Text, nullable=True)
    
    # Generated output (stored as JSON)
    campaign_outline = Column(Text, nullable=True)
    content_checklist_json = Column(Text, nullable=True)
    caption_variants_json = Column(Text, nullable=True)
    metrics_to_track_json = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class MarketingDatabase:
    """Database manager for marketing campaigns with privacy and isolation"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def _get_session(self) -> Session:
        """Get a database session"""
        return self.SessionLocal()
    
    def create_campaign(self, campaign: Campaign) -> Campaign:
        """Create a new campaign"""
        db_campaign = DBCampaign(
            id=campaign.id,
            user_id=campaign.user_id,
            product_service=campaign.product_service,
            target_audience=campaign.target_audience,
            goal=campaign.goal,
            channel=campaign.channel,
            draft_copy=campaign.draft_copy,
            campaign_outline=campaign.campaign_outline,
            content_checklist_json=json.dumps(campaign.content_checklist) if campaign.content_checklist else None,
            caption_variants_json=json.dumps(campaign.caption_variants) if campaign.caption_variants else None,
            metrics_to_track_json=json.dumps(campaign.metrics_to_track) if campaign.metrics_to_track else None,
            created_at=campaign.created_at,
            updated_at=campaign.updated_at
        )
        
        db = self._get_session()
        try:
            db.add(db_campaign)
            db.commit()
            return campaign
        finally:
            db.close()
    
    def get_campaign(self, campaign_id: str, user_id: str) -> Optional[Campaign]:
        """Get a campaign by ID (with user isolation)"""
        db = self._get_session()
        try:
            db_campaign = db.query(DBCampaign).filter(
                DBCampaign.id == campaign_id,
                DBCampaign.user_id == user_id
            ).first()
            
            if not db_campaign:
                return None
            
            return self._db_campaign_to_model(db_campaign)
        finally:
            db.close()
    
    def list_campaigns(self, user_id: str, limit: int = 50) -> List[Campaign]:
        """List campaigns for a user"""
        db = self._get_session()
        try:
            db_campaigns = db.query(DBCampaign).filter(
                DBCampaign.user_id == user_id
            ).order_by(DBCampaign.created_at.desc()).limit(limit).all()
            
            return [self._db_campaign_to_model(c) for c in db_campaigns]
        finally:
            db.close()
    
    def delete_user_data(self, user_id: str) -> int:
        """Delete all marketing data for a user - returns count deleted"""
        db = self._get_session()
        try:
            campaigns = db.query(DBCampaign).filter(DBCampaign.user_id == user_id).all()
            deleted = len(campaigns)
            for c in campaigns:
                db.delete(c)
            db.commit()
            return deleted
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def _db_campaign_to_model(self, db_campaign: DBCampaign) -> Campaign:
        """Convert DB campaign to Pydantic model"""
        content_checklist = json.loads(db_campaign.content_checklist_json) if db_campaign.content_checklist_json else None
        caption_variants = json.loads(db_campaign.caption_variants_json) if db_campaign.caption_variants_json else None
        metrics_to_track = json.loads(db_campaign.metrics_to_track_json) if db_campaign.metrics_to_track_json else None
        
        return Campaign(
            id=db_campaign.id,
            user_id=db_campaign.user_id,
            product_service=db_campaign.product_service,
            target_audience=db_campaign.target_audience,
            goal=db_campaign.goal,
            channel=db_campaign.channel,
            draft_copy=db_campaign.draft_copy,
            campaign_outline=db_campaign.campaign_outline,
            content_checklist=content_checklist,
            caption_variants=caption_variants,
            metrics_to_track=metrics_to_track,
            created_at=db_campaign.created_at,
            updated_at=db_campaign.updated_at
        )


# Global database instance
marketing_db = MarketingDatabase()
