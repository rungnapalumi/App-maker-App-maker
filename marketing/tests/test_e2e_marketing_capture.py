"""
Screenshot capture script for Marketing Agent (uses already-running Streamlit)
Run this after starting: streamlit run aihive_coach/app.py --server.port=8503
"""
import time
import os
from playwright.sync_api import sync_playwright, expect

STREAMLIT_PORT = 8503
BASE_URL = f"http://localhost:{STREAMLIT_PORT}"
SCREENSHOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def capture_screenshots():
    """
    Capture marketing agent screenshots from running Streamlit app
    - Desktop brief form
    - Desktop result with tall viewport to capture full content
    - Mobile result view
    - Workforce hub with Marketing card visible
    """
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        
        # Screenshot 1: Desktop brief form (1280x800)
        print("\n1. Capturing desktop brief form...")
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # Navigate to Marketing Agent via sidebar button
        # Look for the button in the sidebar
        marketing_button = page.locator("button:has-text('📢 Marketing Agent')")
        expect(marketing_button).to_be_visible(timeout=10000)
        marketing_button.click()
        time.sleep(2)
        
        # Verify we're on the campaign page
        expect(page.locator("h1:has-text('📢 Marketing Agent')")).to_be_visible()
        expect(page.locator("text=Campaign Brief")).to_be_visible()
        
        screenshot_path = os.path.join(SCREENSHOT_DIR, "marketing_desktop_brief_form.png")
        page.screenshot(path=screenshot_path, full_page=False)
        print(f"   ✓ Saved: {screenshot_path}")
        
        # Screenshot 2: Desktop result with TALL viewport to capture full analysis
        print("\n2. Generating campaign and capturing full result...")
        page.close()
        
        # Use a tall viewport (2400px) to capture the full result without scrolling
        page = browser.new_page(viewport={"width": 1280, "height": 2400})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # Navigate to Marketing Agent
        marketing_button = page.locator("button:has-text('📢 Marketing Agent')")
        marketing_button.click()
        time.sleep(2)
        
        # Fill in the campaign brief form - wait for form to load
        time.sleep(1)
        
        # Use role-based selectors to avoid help buttons
        page.get_by_role("textbox", name="Product / Service").fill("AIHive Coach")
        page.get_by_role("textbox", name="Target Audience").fill("Professionals seeking personal growth")
        
        # Select goal - click to open dropdown and select option
        page.get_by_role("combobox", name="Campaign Goal").click()
        page.get_by_text("Awareness", exact=True).click()
        
        # Select channel
        page.get_by_role("combobox", name="Primary Channel").click()
        page.get_by_text("LinkedIn", exact=True).click()
        
        # Fill draft copy
        page.get_by_role("textbox", name="Draft Post / Ad Copy (Optional)").fill(
            "Transform your career with AI-powered coaching! 🚀 Get personalized guidance, "
            "actionable insights, and track your growth. Start your free trial today: "
            "https://aihive.com #CareerGrowth #AICoaching #PersonalDevelopment"
        )
        
        # Submit the form - find the button with the generate text
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
        
        print("   ✓ Results verified - all sections present")
        
        # Capture the full result (viewport is tall enough to show everything)
        screenshot_path = os.path.join(SCREENSHOT_DIR, "marketing_desktop_result.png")
        page.screenshot(path=screenshot_path, full_page=False)
        print(f"   ✓ Saved: {screenshot_path}")
        
        page.close()
        
        # Screenshot 3: Mobile result view (390x667 but taller viewport to capture more)
        print("\n3. Capturing mobile result view...")
        page = browser.new_page(viewport={"width": 390, "height": 2400})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # On mobile homepage, find and click the Marketing Agent card's "Open" button
        # Scroll down to find the Marketing card
        page.evaluate("window.scrollBy(0, 1000)")
        time.sleep(1)
        
        # Click the "🚀 Open Marketing Agent" button
        page.locator("button:has-text('🚀 Open Marketing Agent')").click()
        time.sleep(2)
        
        # Fill and submit form on mobile
        time.sleep(1)
        page.get_by_role("textbox", name="Product / Service").fill("AIHive Coach")
        page.get_by_role("textbox", name="Target Audience").fill("Professionals")
        
        # Select goal and channel on mobile
        page.get_by_role("combobox", name="Campaign Goal").click()
        page.get_by_text("Sign-ups", exact=True).click()
        
        page.get_by_role("combobox", name="Primary Channel").click()
        page.get_by_text("Instagram", exact=True).click()
        
        page.get_by_role("textbox", name="Draft Post / Ad Copy (Optional)").fill(
            "Start your growth journey today! 🌱 #PersonalGrowth"
        )
        
        # Generate
        generate_button = page.locator("button:has-text('🚀 Generate Campaign Analysis')")
        generate_button.click()
        time.sleep(5)
        
        # Wait for results
        expect(page.locator("text=📊 Campaign Analysis")).to_be_visible(timeout=10000)
        
        # Capture mobile result
        screenshot_path = os.path.join(SCREENSHOT_DIR, "marketing_mobile_result.png")
        page.screenshot(path=screenshot_path, full_page=False)
        print(f"   ✓ Saved: {screenshot_path}")
        
        page.close()
        
        # Screenshot 4: Workforce hub with Marketing card visible
        print("\n4. Capturing workforce hub with Marketing card...")
        # Use tall viewport to show all cards without scrolling
        page = browser.new_page(viewport={"width": 1280, "height": 2400})
        page.goto(BASE_URL, timeout=30000)
        time.sleep(3)
        
        # Should be on home/hub page by default
        expect(page.locator("text=🏢 AIHive Workforce")).to_be_visible(timeout=5000)
        
        # Verify Marketing card is visible
        expect(page.locator("text=📢 Marketing Agent")).to_be_visible()
        expect(page.locator("text=✅ Demo").first).to_be_visible()
        
        # Capture the hub
        screenshot_path = os.path.join(SCREENSHOT_DIR, "marketing_workforce_hub.png")
        page.screenshot(path=screenshot_path, full_page=False)
        print(f"   ✓ Saved: {screenshot_path}")
        
        page.close()
        browser.close()
    
    print("\n✅ Screenshot capture completed successfully!")
    print("   4 screenshots captured:")
    print("   - Desktop brief form")
    print("   - Desktop result (full campaign analysis)")
    print("   - Mobile result")
    print("   - Workforce hub with Marketing card")


if __name__ == "__main__":
    print("Starting screenshot capture...")
    print(f"Connecting to Streamlit at {BASE_URL}")
    print("Make sure Streamlit is already running!")
    capture_screenshots()
