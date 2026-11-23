"""
Simplified End-to-End tests for AI Workforce Hub
Tests the real running app with proper wait times
"""
import pytest
import sys
import os
import time
from pathlib import Path
from playwright.sync_api import Page, expect

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="module")
def app_url():
    """Get the app URL - must be running"""
    return "http://localhost:8501"


def wait_for_streamlit(page: Page):
    """Wait for Streamlit to fully load"""
    page.wait_for_load_state("networkidle")
    time.sleep(5)  # Extra wait for Streamlit React to render


def test_workforce_hub_loads_and_shows_content(page: Page, app_url: str):
    """Test that the workforce hub page loads with correct content"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    # Check that page loaded
    assert "AIHive" in page.title()
    
    # Check main content is visible
    body_text = page.locator("body").inner_text()
    assert "AI Workforce" in body_text or "AIHive" in body_text
    assert "Coach" in body_text
    assert "Tutor" in body_text


def test_all_five_agents_visible(page: Page, app_url: str):
    """Test that all 5 agents are displayed"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    body_text = page.locator("body").inner_text()
    
    # Check for all 5 agents
    assert "Coach" in body_text
    assert "Tutor" in body_text
    assert "Marketing" in body_text
    assert "Developer" in body_text
    assert "Finance" in body_text


def test_demo_agents_enabled(page: Page, app_url: str):
    """Test that Coach and Tutor have enabled Open buttons"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    # Look for Open buttons (should be 2 for Coach and Tutor)
    buttons = page.locator("button")
    button_texts = [buttons.nth(i).inner_text() for i in range(buttons.count())]
    
    # Should have Open buttons for Coach and Tutor
    open_buttons = [text for text in button_texts if "Open" in text and ("Coach" in text or "Tutor" in text)]
    assert len(open_buttons) >= 2, f"Should have at least 2 Open buttons, found: {open_buttons}"


def test_coming_soon_buttons_disabled(page: Page, app_url: str):
    """Test that coming soon agents have disabled buttons"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    # Look for Coming Soon buttons
    buttons = page.locator("button")
    coming_soon_count = 0
    
    for i in range(buttons.count()):
        button = buttons.nth(i)
        text = button.inner_text()
        if "Coming Soon" in text:
            coming_soon_count += 1
            assert button.is_disabled(), f"Coming Soon button should be disabled: {text}"
    
    assert coming_soon_count == 3, f"Should have 3 Coming Soon buttons, found {coming_soon_count}"


def test_recent_activity_section(page: Page, app_url: str):
    """Test that recent activity section is present"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    body_text = page.locator("body").inner_text()
    assert "Recent Activity" in body_text


def test_demo_mode_warning_present(page: Page, app_url: str):
    """Test that demo mode warning is visible (since no API key is set)"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    body_text = page.locator("body").inner_text()
    assert "Demo Mode" in body_text or "demo mode" in body_text.lower()


def test_finance_has_education_disclaimer(page: Page, app_url: str):
    """Test that Finance agent mentions education/budgeting"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    body_text = page.locator("body").inner_text().lower()
    # Finance agent should mention budgeting or education
    assert "budgeting" in body_text or "education" in body_text


def test_no_placeholder_text_in_page(page: Page, app_url: str):
    """Test that there is no unfinished placeholder text"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    page_content = page.content()
    
    # Check for common placeholder patterns
    assert "[...]" not in page_content
    assert "[PLACEHOLDER]" not in page_content.upper()


def test_capture_desktop_screenshot(page: Page, app_url: str):
    """Capture desktop screenshot of the workforce hub"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    # Set viewport to desktop size
    page.set_viewport_size({"width": 1280, "height": 800})
    time.sleep(2)
    
    # Capture screenshot
    screenshot_path = Path(__file__).parent.parent.parent / "workforce_desktop_hub.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    
    assert screenshot_path.exists()
    # Check file is not empty
    assert screenshot_path.stat().st_size > 1000


def test_capture_mobile_screenshot(page: Page, app_url: str):
    """Capture mobile screenshot of the workforce hub"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    # Set viewport to mobile size (390px wide)
    page.set_viewport_size({"width": 390, "height": 844})
    time.sleep(2)
    
    # Capture screenshot
    screenshot_path = Path(__file__).parent.parent.parent / "workforce_mobile_hub.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    
    assert screenshot_path.exists()
    # Check file is not empty
    assert screenshot_path.stat().st_size > 1000


def test_demo_mode_no_api_key_mentions(page: Page, app_url: str):
    """Test that demo mode page does not mention OPENAI_API_KEY or env vars"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    body_text = page.locator("body").inner_text()
    
    # Should not mention OPENAI_API_KEY
    assert "OPENAI_API_KEY" not in body_text, "Page should not mention OPENAI_API_KEY to end users"
    
    # Should not tell users to add env vars
    assert "Add `" not in body_text or "API" not in body_text, "Page should not instruct users to add env vars"


def test_demo_mode_no_ai_wording_in_titles(page: Page, app_url: str):
    """Test that demo mode page does not use 'AI assistants' or 'AI Team' wording"""
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    body_text = page.locator("body").inner_text()
    
    # In demo mode, should not say "AI assistants" or "AI Team"
    # (Real AI mode can use this wording, but we're testing demo mode here)
    assert "AI assistants" not in body_text, "Demo mode should not say 'AI assistants'"
    assert "Your AI Team" not in body_text, "Demo mode should not say 'Your AI Team'"
    
    # Should use neutral wording instead
    assert "Your AIHive team" in body_text or "Your Team" in body_text, "Should use neutral team wording"
