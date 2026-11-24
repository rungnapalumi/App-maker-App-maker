"""
E2E test for AIHive Marketing Agent using Playwright
Tests that campaign generation produces results
Note: Screenshots are captured by test_e2e_marketing_capture.py
"""
import pytest
import subprocess
import time
import os
import sys
from playwright.sync_api import sync_playwright, expect

# Test configuration
STREAMLIT_PORT = 8504
BASE_URL = f"http://localhost:{STREAMLIT_PORT}"
SCREENSHOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def streamlit_app():
    """Start Streamlit app for testing"""
    # Kill any existing streamlit on this port
    subprocess.run(f"pkill -f 'streamlit.*{STREAMLIT_PORT}'", shell=True)
    time.sleep(1)
    
    # Start Streamlit from the root directory
    app_path = os.path.join(SCREENSHOT_DIR, "aihive_coach", "app.py")
    env = os.environ.copy()
    env['PYTHONPATH'] = SCREENSHOT_DIR  # Ensure root is in path
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path, 
         f"--server.port={STREAMLIT_PORT}", 
         "--server.headless=true",
         "--server.fileWatcherType=none"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=SCREENSHOT_DIR,  # Run from root directory
        env=env
    )
    
    # Wait for app to be ready
    print(f"Starting Streamlit on port {STREAMLIT_PORT}...")
    time.sleep(25)  # Give more time for app to start
    
    yield proc
    
    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)
    subprocess.run(f"pkill -f 'streamlit.*{STREAMLIT_PORT}'", shell=True)


def test_marketing_campaign_generation(streamlit_app):
    """
    E2E test: Verify marketing campaign generation produces correct results
    
    This test ensures that:
    - The Marketing Agent page loads correctly
    - Campaign brief form accepts input
    - Campaign generation produces analysis with all required sections
    - Result content includes actual campaign data (not empty)
    """
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Test campaign generation flow
        print("\nTesting marketing campaign generation...")
        page = browser.new_page(viewport={"width": 1280, "height": 2400})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # Navigate to Marketing Agent
        marketing_button = page.locator("button:has-text('📢 Marketing Agent')")
        expect(marketing_button).to_be_visible(timeout=10000)
        marketing_button.click()
        time.sleep(2)
        
        # Verify we're on the campaign page
        expect(page.locator("h1:has-text('📢 Marketing Agent')")).to_be_visible()
        expect(page.locator("text=Campaign Brief")).to_be_visible()
        
        # Fill in the campaign brief form
        time.sleep(1)
        page.get_by_role("textbox", name="Product / Service").fill("AIHive Coach")
        page.get_by_role("textbox", name="Target Audience").fill("Professionals seeking personal growth")
        
        # Select goal and channel
        page.get_by_role("combobox", name="Campaign Goal").click()
        page.get_by_text("Awareness", exact=True).click()
        
        page.get_by_role("combobox", name="Primary Channel").click()
        page.get_by_text("LinkedIn", exact=True).click()
        
        # Fill draft copy
        page.get_by_role("textbox", name="Draft Post / Ad Copy (Optional)").fill(
            "Transform your career with AI-powered coaching! 🚀 Get personalized guidance, "
            "actionable insights, and track your growth. Start your free trial today: "
            "https://aihive.com #CareerGrowth #AICoaching #PersonalDevelopment"
        )
        
        # Submit the form
        generate_button = page.locator("button:has-text('🚀 Generate Campaign Analysis')")
        generate_button.click()
        time.sleep(5)
        
        # Wait for result section to appear and verify content
        expect(page.locator("text=📊 Campaign Analysis")).to_be_visible(timeout=10000)
        expect(page.locator("text=🎯 Campaign Outline")).to_be_visible(timeout=5000)
        expect(page.locator("text=✅ Content Checklist")).to_be_visible(timeout=5000)
        expect(page.locator("text=💬 Caption Variants")).to_be_visible(timeout=5000)
        expect(page.locator("text=📈 Metrics to Track")).to_be_visible(timeout=5000)
        
        # Assert that actual analysis content is present
        page_content = page.content()
        assert "Objective:" in page_content, "Campaign outline should contain 'Objective:'"
        assert "Word Count" in page_content, "Content checklist should contain 'Word Count'"
        assert "Variant" in page_content, "Should show caption variants"
        assert "AIHive Coach" in page_content, "Should include product name in results"
        assert "Professionals seeking personal growth" in page_content, "Should include audience in results"
        
        print("   ✓ Campaign generation verified - all sections present with correct content")
        
        page.close()
        browser.close()
    
    print("\n✅ E2E test completed successfully!")


if __name__ == "__main__":
    # Can be run directly for screenshot capture
    with sync_playwright() as p:
        # Start streamlit manually for direct run
        print("Note: When running directly, start streamlit manually first:")
        print(f"  streamlit run aihive_coach/app.py --server.port={STREAMLIT_PORT}")
        print("\nThen run this script.")
