"""
Unit tests for the agent registry
"""
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from workforce.registry import (
    get_all_agents,
    get_agent_by_id,
    get_enabled_agents,
    Agent
)


def test_get_all_agents():
    """Test getting all agents"""
    agents = get_all_agents()
    
    assert len(agents) >= 5, "Should have at least 5 agents"
    assert all(isinstance(agent, Agent) for agent in agents), "All items should be Agent instances"


def test_agent_structure():
    """Test agent structure has required fields"""
    agents = get_all_agents()
    
    for agent in agents:
        assert hasattr(agent, 'id'), "Agent must have id"
        assert hasattr(agent, 'name'), "Agent must have name"
        assert hasattr(agent, 'role'), "Agent must have role"
        assert hasattr(agent, 'status'), "Agent must have status"
        assert hasattr(agent, 'enabled'), "Agent must have enabled"
        
        # Validate status values
        assert agent.status in ["Demo", "Coming soon"], f"Invalid status: {agent.status}"
        
        # Validate enabled is boolean
        assert isinstance(agent.enabled, bool), "enabled must be boolean"


def test_get_agent_by_id():
    """Test getting agent by ID"""
    # Test valid IDs
    coach = get_agent_by_id("coach")
    assert coach is not None, "Coach agent should exist"
    assert coach.id == "coach"
    assert coach.name == "AIHive Coach"
    
    tutor = get_agent_by_id("tutor")
    assert tutor is not None, "Tutor agent should exist"
    assert tutor.id == "tutor"
    assert tutor.name == "AIHive Tutor"
    
    # Test invalid ID
    invalid = get_agent_by_id("nonexistent")
    assert invalid is None, "Should return None for invalid ID"


def test_get_enabled_agents():
    """Test getting only enabled agents"""
    enabled_agents = get_enabled_agents()
    
    # All enabled agents should have enabled=True
    assert all(agent.enabled for agent in enabled_agents), "All returned agents should be enabled"
    
    # Coach and Tutor should be enabled
    enabled_ids = [agent.id for agent in enabled_agents]
    assert "coach" in enabled_ids, "Coach should be enabled"
    assert "tutor" in enabled_ids, "Tutor should be enabled"


def test_coach_agent_details():
    """Test Coach agent specific details"""
    coach = get_agent_by_id("coach")
    
    assert coach.status == "Demo", "Coach should have Demo status"
    assert coach.enabled is True, "Coach should be enabled"
    assert coach.navigation_target == "AIHive Coach", "Coach should have correct navigation target"


def test_tutor_agent_details():
    """Test Tutor agent specific details"""
    tutor = get_agent_by_id("tutor")
    
    assert tutor.status == "Demo", "Tutor should have Demo status"
    assert tutor.enabled is True, "Tutor should be enabled"
    assert tutor.navigation_target == "Start Learning", "Tutor should have correct navigation target"


def test_coming_soon_agents():
    """Test coming soon agents are disabled"""
    all_agents = get_all_agents()
    
    coming_soon_agents = [agent for agent in all_agents if agent.status == "Coming soon"]
    
    # All coming soon agents should be disabled
    assert all(not agent.enabled for agent in coming_soon_agents), \
        "All 'Coming soon' agents should be disabled"
    
    # Check specific coming soon agents
    marketing = get_agent_by_id("marketing")
    assert marketing is not None, "Marketing agent should exist"
    assert marketing.status == "Coming soon"
    assert marketing.enabled is False
    
    developer = get_agent_by_id("developer")
    assert developer is not None, "Developer agent should exist"
    assert developer.status == "Coming soon"
    assert developer.enabled is False
    
    finance = get_agent_by_id("finance")
    assert finance is not None, "Finance agent should exist"
    assert finance.status == "Coming soon"
    assert finance.enabled is False
    # Verify finance disclaimer in role
    assert "education" in finance.role.lower() or "budgeting" in finance.role.lower(), \
        "Finance agent should mention education/budgeting"


def test_agent_uniqueness():
    """Test all agent IDs are unique"""
    agents = get_all_agents()
    ids = [agent.id for agent in agents]
    
    assert len(ids) == len(set(ids)), "All agent IDs should be unique"


def test_registry_is_single_source():
    """Test registry is single source of truth"""
    # This test verifies the registry module exports a single AGENTS list
    from workforce import registry
    
    assert hasattr(registry, 'AGENTS'), "Registry should have AGENTS list"
    assert isinstance(registry.AGENTS, list), "AGENTS should be a list"
    assert len(registry.AGENTS) >= 5, "AGENTS should have at least 5 entries"
