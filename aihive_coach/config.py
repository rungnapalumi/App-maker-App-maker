"""
AIHive Coach Configuration
Manages environment variables and app settings
"""
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""
    
    # AI Provider Settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE")  # For compatible APIs
    
    # App Settings
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "aihive_coach.db")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Feature Flags
    ENABLE_DEMO_MODE: bool = os.getenv("ENABLE_DEMO_MODE", "auto").lower() in ["true", "auto"]
    LOG_API_USAGE: bool = os.getenv("LOG_API_USAGE", "true").lower() == "true"
    
    # Crisis Resources
    THAILAND_MENTAL_HEALTH_HOTLINE: str = "1323"
    INTERNATIONAL_CRISIS_TEXT: str = "Crisis Text Line (US): Text HOME to 741741"
    
    @classmethod
    def is_demo_mode(cls) -> bool:
        """Check if app should run in demo mode (no real AI API)"""
        if not cls.ENABLE_DEMO_MODE:
            return False
        # Auto mode: use demo if no API key
        if cls.OPENAI_API_KEY:
            return False
        return True
    
    @classmethod
    def get_model_info(cls) -> str:
        """Return human-readable model info"""
        if cls.is_demo_mode():
            return "Demo Mode (Scripted Responses)"
        return f"{cls.OPENAI_MODEL}"


config = Config()
