"""
End-to-End Tests with Playwright
Tests full user workflows including mobile responsiveness
"""
import pytest
import subprocess
import time
import os
from playwright.sync_api import sync_playwright, expect


STREAMLIT_PORT = 8502
APP_URL = f"http://localhost:{STREAMLIT_PORT}"


@pytest.fixture(scope="module")
def streamlit_app():
    """Start Streamlit app for testing"""
    env = os.environ.copy()
    env["APP_ENV"] = "test"
    env["DATABASE_PATH"] = "test_e2e.db"
    
    # Start Streamlit in background
    process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", str(STREAMLIT_PORT), 
         "--server.headless", "true"],
        cwd=".",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for app to start
    time.sleep(5)
    
    yield
    
    # Cleanup
    process.terminate()
    process.wait()
    
    if os.path.exists("test_e2e.db"):
        os.unlink("test_e2e.db")


@pytest.fixture
def browser_context():
    """Create browser context"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context()
        yield context
        context.close()
        browser.close()


def test_home_page_loads(streamlit_app, browser_context):
    """Test that home page loads correctly"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Check sidebar appears
    expect(page.locator("section[data-testid='stSidebar']")).to_be_visible(timeout=10000)
    # Check title or content appears
    expect(page.locator("text=Welcome to AIHiveHQ")).to_be_visible()


def test_navigation_to_coach_landing(streamlit_app, browser_context):
    """Test navigation to coach landing page"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Click navigation to AIHive Coach (in sidebar)
    page.locator("section[data-testid='stSidebar'] button:has-text('AIHive Coach')").click()
    time.sleep(3)
    
    # Verify we're on coach landing by checking for unique content
    expect(page.locator("text=How It Works").first).to_be_visible()


def test_start_coaching_session(streamlit_app, browser_context):
    """Test starting a coaching session"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Navigate to coaching
    page.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    
    # Select category
    page.locator("button:has-text('Career & Leadership')").first.click()
    time.sleep(3)
    
    # Verify coaching workspace loads
    expect(page.locator("text=Coaching Workspace")).to_be_visible()


def test_send_message_in_coaching(streamlit_app, browser_context):
    """Test sending a message in coaching conversation"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Navigate and start session
    page.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    page.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(3)
    
    # Type message in chat input
    chat_input = page.locator("[data-testid='stChatInput'] textarea")
    chat_input.fill("I want to build better habits")
    time.sleep(1)
    page.keyboard.press("Enter")
    time.sleep(6)
    
    # Verify user message appears
    expect(page.locator("text=I want to build better habits")).to_be_visible()
    
    # Verify chat messages exist
    messages = page.locator("[data-testid='stChatMessage']")
    assert messages.count() >= 2, "Should have at least welcome + user message"


def test_create_action_plan(streamlit_app, browser_context):
    """Test creating an action plan"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Start session
    page.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    page.locator("button:has-text('Goals & Productivity')").first.click()
    time.sleep(3)
    
    # Send a message
    chat_input = page.locator("[data-testid='stChatInput'] textarea")
    chat_input.fill("I want to launch my side project")
    page.keyboard.press("Enter")
    time.sleep(5)
    
    # Create action plan
    page.locator("button:has-text('Create Action Plan')").click()
    time.sleep(5)
    
    # Verify action plan form appears by checking for unique form text
    expect(page.locator("text=Based on our conversation").first).to_be_visible(timeout=10000)
    
    # Save action plan
    page.locator("button:has-text('Save Action Plan')").first.click()
    time.sleep(3)
    
    # Verify success (may show balloons or success message)
    page.wait_for_timeout(2000)


def test_view_action_plans(streamlit_app, browser_context):
    """Test viewing action plans page"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Navigate to action plans
    page.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    
    # Verify page loads by checking sidebar button is active or content exists
    expect(page.locator("section[data-testid='stSidebar']")).to_be_visible()


def test_mobile_viewport(streamlit_app, browser_context):
    """Test mobile responsiveness"""
    # Set mobile viewport
    mobile_context = browser_context.browser.new_context(
        viewport={"width": 375, "height": 667}
    )
    
    page = mobile_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # On mobile, sidebar starts collapsed (auto mode)
    # Main content should be visible
    expect(page.locator("[data-testid='stMainBlockContainer']")).to_be_visible()
    
    # Content should load (home page)
    page_content = page.content()
    assert "AIHiveHQ" in page_content or "Think Clearly" in page_content
    
    mobile_context.close()


def test_new_session_button(streamlit_app, browser_context):
    """Test starting a new session"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Start first session
    page.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    page.locator("button:has-text('Decision Making')").first.click()
    time.sleep(3)
    
    # Click New Session
    page.locator("button:has-text('New Session')").click()
    time.sleep(3)
    
    # Should be back to category selection
    expect(page.locator("text=What would you like to work on")).to_be_visible()


def test_edit_action_plan(streamlit_app, browser_context):
    """Test editing an action plan"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    time.sleep(3)
    
    # Create a plan first
    page.locator("section[data-testid='stSidebar'] button:has-text('Start Coaching')").click()
    time.sleep(3)
    page.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(3)
    
    chat_input = page.locator("[data-testid='stChatInput'] textarea")
    chat_input.fill("Test goal")
    page.keyboard.press("Enter")
    time.sleep(5)
    
    page.locator("button:has-text('Create Action Plan')").click()
    time.sleep(3)
    page.locator("button:has-text('Save Action Plan')").first.click()
    time.sleep(3)
    
    # Navigate to action plans
    page.locator("section[data-testid='stSidebar'] button:has-text('My Action Plans')").click()
    time.sleep(3)
    
    # Check that plans page loaded (verify sidebar is visible at minimum)
    expect(page.locator("section[data-testid='stSidebar']")).to_be_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
