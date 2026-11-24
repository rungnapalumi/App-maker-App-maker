"""
Tests for Marketing Database
Ensures proper user isolation and data deletion
"""
import pytest
import tempfile
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from marketing.database import MarketingDatabase
from marketing.models import Campaign


@pytest.fixture
def test_db():
    """Create a temporary test database"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = MarketingDatabase(db_path=db_path)
    yield db
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


def test_user_isolation_campaigns(test_db):
    """Ensure campaigns are properly isolated per user"""
    # Create campaigns for user1
    campaign1_user1 = Campaign(
        user_id="user1",
        product_service="Product A",
        target_audience="Audience A",
        goal="Awareness",
        channel="Facebook"
    )
    
    campaign2_user1 = Campaign(
        user_id="user1",
        product_service="Product B",
        target_audience="Audience B",
        goal="Sales",
        channel="Instagram"
    )
    
    # Create campaigns for user2
    campaign1_user2 = Campaign(
        user_id="user2",
        product_service="Product C",
        target_audience="Audience C",
        goal="Sign-ups",
        channel="Email"
    )
    
    # Save all campaigns
    test_db.create_campaign(campaign1_user1)
    test_db.create_campaign(campaign2_user1)
    test_db.create_campaign(campaign1_user2)
    
    # List campaigns for user1
    user1_campaigns = test_db.list_campaigns("user1")
    assert len(user1_campaigns) == 2
    assert all(c.user_id == "user1" for c in user1_campaigns)
    assert any(c.product_service == "Product A" for c in user1_campaigns)
    assert any(c.product_service == "Product B" for c in user1_campaigns)
    
    # List campaigns for user2
    user2_campaigns = test_db.list_campaigns("user2")
    assert len(user2_campaigns) == 1
    assert all(c.user_id == "user2" for c in user2_campaigns)
    assert user2_campaigns[0].product_service == "Product C"
    
    # Verify user1 cannot access user2's campaign
    retrieved = test_db.get_campaign(campaign1_user2.id, "user1")
    assert retrieved is None
    
    # Verify user2 can access their own campaign
    retrieved = test_db.get_campaign(campaign1_user2.id, "user2")
    assert retrieved is not None
    assert retrieved.product_service == "Product C"


def test_delete_user_data(test_db):
    """Test that delete_user_data properly removes all user campaigns"""
    # Create campaigns for multiple users
    campaign1 = Campaign(
        user_id="user_to_delete",
        product_service="Product 1",
        target_audience="Audience 1",
        goal="Awareness",
        channel="Facebook"
    )
    
    campaign2 = Campaign(
        user_id="user_to_delete",
        product_service="Product 2",
        target_audience="Audience 2",
        goal="Sales",
        channel="Instagram"
    )
    
    campaign3 = Campaign(
        user_id="user_to_keep",
        product_service="Product 3",
        target_audience="Audience 3",
        goal="Sign-ups",
        channel="Email"
    )
    
    test_db.create_campaign(campaign1)
    test_db.create_campaign(campaign2)
    test_db.create_campaign(campaign3)
    
    # Verify campaigns exist
    assert len(test_db.list_campaigns("user_to_delete")) == 2
    assert len(test_db.list_campaigns("user_to_keep")) == 1
    
    # Delete user_to_delete's data
    deleted_count = test_db.delete_user_data("user_to_delete")
    assert deleted_count == 2
    
    # Verify user_to_delete's campaigns are gone
    assert len(test_db.list_campaigns("user_to_delete")) == 0
    
    # Verify user_to_keep's campaigns are intact
    remaining = test_db.list_campaigns("user_to_keep")
    assert len(remaining) == 1
    assert remaining[0].product_service == "Product 3"


def test_campaign_persistence(test_db):
    """Test that campaigns are properly saved and retrieved"""
    campaign = Campaign(
        user_id="test_user",
        product_service="Test Product",
        target_audience="Test Audience",
        goal="Awareness",
        channel="Facebook",
        draft_copy="Test draft copy",
        campaign_outline="Test outline",
        content_checklist={"word_count": 10, "has_cta": True},
        caption_variants=["Variant 1", "Variant 2"],
        metrics_to_track=["Metric 1", "Metric 2"]
    )
    
    # Save campaign
    test_db.create_campaign(campaign)
    
    # Retrieve campaign
    retrieved = test_db.get_campaign(campaign.id, "test_user")
    
    # Verify all fields
    assert retrieved is not None
    assert retrieved.id == campaign.id
    assert retrieved.user_id == "test_user"
    assert retrieved.product_service == "Test Product"
    assert retrieved.target_audience == "Test Audience"
    assert retrieved.goal == "Awareness"
    assert retrieved.channel == "Facebook"
    assert retrieved.draft_copy == "Test draft copy"
    assert retrieved.campaign_outline == "Test outline"
    assert retrieved.content_checklist["word_count"] == 10
    assert retrieved.content_checklist["has_cta"] == True
    assert len(retrieved.caption_variants) == 2
    assert "Variant 1" in retrieved.caption_variants
    assert len(retrieved.metrics_to_track) == 2


def test_list_campaigns_limit(test_db):
    """Test that list_campaigns respects the limit parameter"""
    # Create many campaigns for one user
    for i in range(10):
        campaign = Campaign(
            user_id="test_user",
            product_service=f"Product {i}",
            target_audience="Audience",
            goal="Awareness",
            channel="Facebook"
        )
        test_db.create_campaign(campaign)
    
    # List with limit
    campaigns = test_db.list_campaigns("test_user", limit=5)
    assert len(campaigns) == 5
    
    # List all
    campaigns_all = test_db.list_campaigns("test_user", limit=50)
    assert len(campaigns_all) == 10


def test_list_campaigns_order(test_db):
    """Test that list_campaigns returns newest first"""
    import time
    
    # Create campaigns with slight time gaps
    campaign1 = Campaign(
        user_id="test_user",
        product_service="First",
        target_audience="Audience",
        goal="Awareness",
        channel="Facebook"
    )
    test_db.create_campaign(campaign1)
    
    time.sleep(0.01)  # Small delay
    
    campaign2 = Campaign(
        user_id="test_user",
        product_service="Second",
        target_audience="Audience",
        goal="Sales",
        channel="Instagram"
    )
    test_db.create_campaign(campaign2)
    
    # List campaigns
    campaigns = test_db.list_campaigns("test_user")
    
    # Newest should be first
    assert campaigns[0].product_service == "Second"
    assert campaigns[1].product_service == "First"


def test_nonexistent_campaign(test_db):
    """Test retrieving a nonexistent campaign returns None"""
    retrieved = test_db.get_campaign("nonexistent_id", "test_user")
    assert retrieved is None


def test_empty_list_for_new_user(test_db):
    """Test that a new user has no campaigns"""
    campaigns = test_db.list_campaigns("new_user")
    assert len(campaigns) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
