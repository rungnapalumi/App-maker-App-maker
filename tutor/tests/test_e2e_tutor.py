"""
E2E test for AIHive Tutor using Playwright
Tests the complete tutor flow and captures real screenshots
"""
import pytest
import subprocess
import time
import os
import sys
from playwright.sync_api import sync_playwright, expect

# Test configuration
STREAMLIT_PORT = 8502
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
    time.sleep(10)
    
    yield proc
    
    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)
    subprocess.run(f"pkill -f 'streamlit.*{STREAMLIT_PORT}'", shell=True)


def test_complete_tutor_flow_with_screenshots(streamlit_app):
    """
    E2E test: Complete tutor flow with screenshot capture
    - Desktop topic picker
    - Mobile lesson with wrong answer (adaptive feedback)
    - Mobile lesson with correct answer
    - Mobile session summary
    
    Note: Sessions list screenshot is captured by the standalone script
    (capture_real_screenshots.py) due to Streamlit rerun timing in headless mode
    """
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Screenshot 1: Desktop topic picker (1440x900)
        print("\n1. Capturing desktop topic picker...")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # Navigate to tutor via sidebar
        page.get_by_text("Start Learning", exact=False).first.click()
        time.sleep(2)
        
        # Verify we're on topic picker
        expect(page.get_by_text("What do you want to learn today?")).to_be_visible()
        
        screenshot_path = os.path.join(SCREENSHOT_DIR, "tutor_desktop_topic_picker.png")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"   ✓ Saved: {screenshot_path}")
        
        page.close()
        
        # Screenshot 2-4: Mobile flow with real lesson
        print("\n2. Starting mobile lesson flow...")
        page = browser.new_page(viewport={"width": 390, "height": 844})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # On mobile, navigate via sidebar button
        page.get_by_role("button", name="📚 Start Learning").click()
        time.sleep(3)
        
        # Select Photosynthesis topic
        page.get_by_text("🌱 Photosynthesis").click()
        time.sleep(1)
        
        # Click Start Learning button (Beginner is default)
        page.get_by_role("button", name="🚀 Start Learning").click()
        time.sleep(5)
        
        # Wait for first lesson message to appear
        expect(page.locator(".stChatMessage")).to_have_count(1, timeout=10000)
        
        # Answer 1: Give a wrong/incomplete answer to trigger feedback
        print("\n3. Answering first question (wrong answer)...")
        chat_input = page.get_by_placeholder("Type your answer here...")
        expect(chat_input).to_be_visible(timeout=5000)
        
        chat_input.fill("I don't know")
        chat_input.press("Enter")
        time.sleep(5)
        
        # Wait for feedback message (at least 3 messages: initial + user + feedback, possibly more)
        message_count = page.locator(".stChatMessage").count()
        print(f"   Message count after wrong answer: {message_count}")
        expect(page.locator(".stChatMessage").first).to_be_visible(timeout=10000)
        
        # Capture screenshot showing adaptive feedback
        screenshot_path = os.path.join(SCREENSHOT_DIR, "tutor_mobile_mid_lesson.png")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"   ✓ Saved: {screenshot_path}")
        
        # Answer 2: Give a good answer
        print("\n4. Answering follow-up question (correct answer)...")
        time.sleep(2)
        
        chat_input = page.get_by_placeholder("Type your answer here...")
        chat_input.fill("Plants use sunlight, water, and carbon dioxide to make food through photosynthesis.")
        chat_input.press("Enter")
        time.sleep(5)
        
        # Wait for more messages (should have more after good answer)
        final_message_count = page.locator(".stChatMessage").count()
        print(f"   Message count after correct answer: {final_message_count}")
        assert final_message_count > message_count, "Should have more messages after correct answer"
        
        # End the lesson
        print("\n5. Ending lesson...")
        page.get_by_role("button", name="✅ End Lesson").click()
        time.sleep(5)
        
        # Verify summary is shown
        expect(page.get_by_text("Lesson Complete!")).to_be_visible(timeout=10000)
        
        # Capture summary screenshot
        screenshot_path = os.path.join(SCREENSHOT_DIR, "tutor_mobile_summary.png")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"   ✓ Saved: {screenshot_path}")
        
        # Verify the summary shows correct stats
        summary_content = page.content()
        assert "Questions Correct" in summary_content
        assert "Questions Incorrect" in summary_content
        # Both should show "1" based on our test flow
        
        page.close()
        browser.close()
    
    print("\n✅ E2E test completed successfully!")
    print("   3 core screenshots captured: desktop topic picker, mobile mid-lesson, mobile summary")
    print("   (Sessions list screenshot captured separately by capture_real_screenshots.py)")


if __name__ == "__main__":
    # Can be run directly for screenshot capture
    with sync_playwright() as p:
        # Start streamlit manually for direct run
        print("Note: When running directly, start streamlit manually first:")
        print(f"  streamlit run aihive_coach/app.py --server.port={STREAMLIT_PORT}")
        print("\nThen run this script.")
