"""
AIHive Agent Registry
Single source of truth for all AIHive agents
"""
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class Agent:
    """Represents an AIHive agent"""
    id: str
    name: str
    role: str  # One-line description
    status: Literal["Demo", "Coming soon"]
    enabled: bool
    navigation_target: str = None  # Streamlit navigation page name


# Agent Registry - Add new agents here
AGENTS: List[Agent] = [
    Agent(
        id="coach",
        name="AIHive Coach",
        role="Think clearly. Act confidently. Grow daily.",
        status="Demo",
        enabled=True,
        navigation_target="AIHive Coach"
    ),
    Agent(
        id="tutor",
        name="AIHive Tutor",
        role="Learn step-by-step with adaptive feedback.",
        status="Demo",
        enabled=True,
        navigation_target="Start Learning"
    ),
    Agent(
        id="marketing",
        name="Marketing Assistant",
        role="Content strategy and campaign planning.",
        status="Coming soon",
        enabled=False,
        navigation_target=None
    ),
    Agent(
        id="developer",
        name="Developer Assistant",
        role="Code review, debugging, and architecture guidance.",
        status="Coming soon",
        enabled=False,
        navigation_target=None
    ),
    Agent(
        id="finance",
        name="Finance Assistant",
        role="Personal budgeting and expense tracking (education only).",
        status="Coming soon",
        enabled=False,
        navigation_target=None
    ),
]


def get_all_agents() -> List[Agent]:
    """Get all registered agents"""
    return AGENTS


def get_agent_by_id(agent_id: str) -> Agent:
    """Get agent by ID"""
    for agent in AGENTS:
        if agent.id == agent_id:
            return agent
    return None


def get_enabled_agents() -> List[Agent]:
    """Get only enabled agents"""
    return [agent for agent in AGENTS if agent.enabled]
