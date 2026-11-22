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
    
    # Check title appears
    expect(page.locator("text=Welcome to AIHiveHQ")).to_be_visible(timeout=10000)
    expect(page.locator("text=Think Clearly")).to_be_visible()


def test_navigation_to_coach_landing(streamlit_app, browser_context):
    """Test navigation to coach landing page"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Click navigation to AIHive Coach
    page.click("text=AIHive Coach")
    time.sleep(2)
    
    # Verify we're on coach landing
    expect(page.locator("text=How It Works")).to_be_visible()


def test_start_coaching_session(streamlit_app, browser_context):
    """Test starting a coaching session"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Navigate to coaching
    page.click("text=Start Coaching")
    time.sleep(2)
    
    # Select category
    page.click("text=Career & Leadership")
    time.sleep(2)
    
    # Verify coaching workspace loads
    expect(page.locator("text=Coaching Workspace")).to_be_visible()
    expect(page.locator("text=What would you like to work on")).to_be_visible()


def test_send_message_in_coaching(streamlit_app, browser_context):
    """Test sending a message in coaching conversation"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Navigate and start session
    page.click("text=Start Coaching")
    time.sleep(2)
    page.click("text=Personal Growth")
    time.sleep(2)
    
    # Type message
    page.fill("textarea", "I want to build better habits")
    time.sleep(1)
    
    # Send message
    page.click("button:has-text('Send')")
    time.sleep(3)
    
    # Verify user message appears
    expect(page.locator("text=I want to build better habits")).to_be_visible()
    
    # Verify coach response appears (demo mode)
    # Should have a response with a question
    page.wait_for_selector(".assistant-message", timeout=5000)


def test_create_action_plan(streamlit_app, browser_context):
    """Test creating an action plan"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Start session and have conversation
    page.click("text=Start Coaching")
    time.sleep(2)
    page.click("text=Goals & Productivity")
    time.sleep(2)
    
    # Send a message
    page.fill("textarea", "I want to launch my side project")
    page.click("button:has-text('Send')")
    time.sleep(3)
    
    # Create action plan
    page.click("text=Create Action Plan")
    time.sleep(2)
    
    # Verify action plan form appears
    expect(page.locator("text=Create Your Action Plan")).to_be_visible()
    expect(page.locator("text=My Goal")).to_be_visible()
    
    # Edit fields
    page.fill("textarea[value*='launch']", "Launch MVP by end of month")
    time.sleep(1)
    
    # Save action plan
    page.click("button:has-text('Save Action Plan')")
    time.sleep(2)
    
    # Verify success
    expect(page.locator("text=Action Plan saved successfully")).to_be_visible()


def test_view_action_plans(streamlit_app, browser_context):
    """Test viewing action plans page"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Navigate to action plans
    page.click("text=My Action Plans")
    time.sleep(2)
    
    # Either see empty state or existing plans
    # (depends on if previous tests ran)
    page_content = page.content()
    assert "My Action Plans" in page_content


def test_mobile_viewport(streamlit_app, browser_context):
    """Test mobile responsiveness"""
    # Set mobile viewport
    mobile_context = browser_context.browser.new_context(
        viewport={"width": 375, "height": 667},
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
    )
    
    page = mobile_context.new_page()
    page.goto(APP_URL)
    
    # Check that page loads on mobile
    expect(page.locator("text=AIHiveHQ")).to_be_visible()
    
    # Check coaching page on mobile
    page.click("text=Start Coaching")
    time.sleep(2)
    
    # Category buttons should be visible
    expect(page.locator("text=Career & Leadership")).to_be_visible()
    
    mobile_context.close()


def test_new_session_button(streamlit_app, browser_context):
    """Test starting a new session"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Start first session
    page.click("text=Start Coaching")
    time.sleep(2)
    page.click("text=Decision Making")
    time.sleep(2)
    
    # Verify session started
    expect(page.locator("text=Decision Making")).to_be_visible()
    
    # Click New Session
    page.click("text=New Session")
    time.sleep(2)
    
    # Should be back to category selection
    expect(page.locator("text=What would you like to work on")).to_be_visible()


def test_edit_action_plan(streamlit_app, browser_context):
    """Test editing an action plan"""
    page = browser_context.new_page()
    page.goto(APP_URL)
    
    # Create a plan first
    page.click("text=Start Coaching")
    time.sleep(2)
    page.click("text=Personal Growth")
    time.sleep(2)
    
    page.fill("textarea", "Test goal")
    page.click("button:has-text('Send')")
    time.sleep(3)
    
    page.click("text=Create Action Plan")
    time.sleep(2)
    page.click("button:has-text('Save Action Plan')")
    time.sleep(2)
    
    # Navigate to action plans
    page.click("text=My Action Plans")
    time.sleep(2)
    
    # Expand first plan
    page.click("summary >> nth=0")
    time.sleep(1)
    
    # Click edit
    page.click("button:has-text('Edit')")
    time.sleep(1)
    
    # Verify edit form appears
    expect(page.locator("textarea").first).to_be_visible()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
