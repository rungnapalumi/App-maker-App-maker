"""
E2E test for activity panel functionality
Tests that sessions created in the browser appear in the activity panel
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
    """Get the app URL"""
    return "http://localhost:8501"


def wait_for_streamlit(page: Page):
    """Wait for Streamlit to fully load"""
    page.wait_for_load_state("networkidle")
    time.sleep(5)


def test_activity_panel_shows_created_sessions(page: Page, app_url: str):
    """
    Test that sessions created in the same browser session appear in activity panel.
    This is a critical user flow test.
    """
    # Start at hub
    page.goto(app_url, wait_until="networkidle", timeout=60000)
    wait_for_streamlit(page)
    
    # Initial state: activity panel should show "No activity yet"
    body_text = page.locator("body").inner_text()
    assert "My Recent Activity" in body_text, "Activity panel heading should be visible"
    assert "No activity yet" in body_text, "Should show empty state initially"
    
    # Navigate to Coach to create a session
    coach_button = page.locator("button:has-text('Open AIHive Coach')").first
    expect(coach_button).to_be_visible()
    coach_button.click()
    wait_for_streamlit(page)
    
    # Verify we're on coach page
    body_text = page.locator("body").inner_text()
    assert "AIHive Coach" in body_text or "Think Clearly" in body_text
    
    # Click a category to start a session (creates a session in DB)
    # Try to find and click Personal Growth or any category button
    try:
        category_button = page.locator("button:has-text('Personal Growth')").first
        if category_button.is_visible(timeout=5000):
            category_button.click()
            wait_for_streamlit(page)
    except:
        # If category buttons not found, just continue
        # The session might be auto-created on page load
        pass
    
    # Return to hub
    # Open sidebar if needed
    try:
        sidebar_buttons = page.locator("button[kind='header']")
        if sidebar_buttons.count() > 0:
            sidebar_buttons.first.click()
            time.sleep(1)
    except:
        pass
    
    # Click hub button
    hub_button = page.locator("button:has-text('Workforce Hub')")
    if hub_button.count() > 0:
        hub_button.first.click()
        wait_for_streamlit(page)
    else:
        # Try alternative navigation
        home_button = page.locator("button:has-text('Home')").or_(
            page.locator("button:has-text('🏠')")
        )
        if home_button.count() > 0:
            home_button.first.click()
            wait_for_streamlit(page)
    
    # NOW check activity panel - it should show the session
    body_text = page.locator("body").inner_text()
    
    # This is the key assertion: activity panel should NO LONGER say "No activity yet"
    # It should either:
    # 1. Show "AIHive Coach" in an activity entry
    # 2. Show "Personal Growth" (the category)
    # 3. Show some session data
    
    # For now, let's check if the empty state is gone
    if "No activity yet" in body_text:
        # This is a BUG - session was created but not showing
        pytest.fail(
            "BUG: Activity panel still shows 'No activity yet' after creating a coach session. "
            "Sessions created in the browser should appear in the activity panel."
        )
    
    # Verify activity entry exists
    # Should see "AIHive Coach" as an agent name in an activity entry
    assert "AIHive Coach" in body_text or "Coach" in body_text, \
        "Activity panel should show coach session"
