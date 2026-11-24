"""
Screenshot capture for Marketing Agent
Captures desktop and mobile views
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
SCREENSHOTS_DIR = "/agent/repos/App-maker-App-maker"

def capture_marketing_screenshots():
    """Capture all marketing agent screenshots"""
    with sync_playwright() as p:
        # Desktop screenshots
        browser = p.chromium.launch(headless=True)
        
        # Desktop context
        desktop = browser.new_context(viewport={"width": 1280, "height": 800})
        page_desktop = desktop.new_page()
        
        print("Navigating to app...")
        page_desktop.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        time.sleep(3)
        
        # Navigate to Marketing Agent
        print("Opening Marketing Agent...")
        page_desktop.get_by_text("Marketing Agent", exact=False).first.click()
        time.sleep(2)
        
        # Screenshot 1: Desktop brief form
        print("Capturing desktop brief form...")
        page_desktop.screenshot(path=f"{SCREENSHOTS_DIR}/marketing_desktop_brief_form.png", full_page=True)
        
        # Fill in the form
        print("Filling form...")
        try:
            inputs = page_desktop.query_selector_all("input[type='text']")
            if len(inputs) >= 2:
                inputs[0].fill("EcoBottle - Sustainable Water Bottle")
                time.sleep(0.5)
                inputs[1].fill("Environmentally conscious millennials aged 25-35")
                time.sleep(0.5)
            
            # Find selects and pick options
            selects = page_desktop.query_selector_all("select")
            if len(selects) >= 2:
                # Goal select
                selects[0].select_option("Awareness")
                time.sleep(0.5)
                # Channel select
                selects[1].select_option("Instagram")
                time.sleep(0.5)
            
            # Add draft copy
            draft = "🌱 Meet EcoBottle! The sustainable water bottle that keeps drinks cold for 24hrs. Made from recycled materials. Join the eco revolution! Shop now: ecobottle.com #sustainable #ecofriendly #zerowaste"
            textarea = page_desktop.query_selector("textarea")
            if textarea:
                textarea.fill(draft)
                time.sleep(0.5)
            
            # Submit
            print("Generating campaign analysis...")
            submit_button = page_desktop.query_selector("button:has-text('Generate Campaign Analysis')")
            if submit_button:
                submit_button.click()
                time.sleep(6)
        except Exception as e:
            print(f"Note: Form interaction had issues: {e}")
            print("Capturing what we have so far...")
        
        # Screenshot 2: Desktop result
        print("Capturing desktop result...")
        page_desktop.screenshot(path=f"{SCREENSHOTS_DIR}/marketing_desktop_result.png", full_page=True)
        
        desktop.close()
        
        # Mobile screenshots
        mobile = browser.new_context(
            viewport={"width": 375, "height": 667},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        )
        page_mobile = mobile.new_page()
        
        print("Navigating to app (mobile)...")
        page_mobile.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        time.sleep(3)
        
        # Open sidebar  
        print("Opening sidebar (mobile)...")
        try:
            # Try multiple ways to open sidebar
            sidebar_buttons = page_mobile.query_selector_all("button")
            if sidebar_buttons:
                sidebar_buttons[0].click()  # First button is usually the sidebar toggle
                time.sleep(2)
        except Exception as e:
            print(f"Sidebar open attempt: {e}")
        
        # Navigate to Marketing Agent
        print("Opening Marketing Agent (mobile)...")
        try:
            marketing_link = page_mobile.query_selector("text=📢 Marketing Agent")
            if marketing_link:
                # Scroll into view first
                marketing_link.scroll_into_view_if_needed()
                time.sleep(1)
                marketing_link.click()
                time.sleep(2)
            else:
                print("Could not find Marketing Agent link on mobile, skipping mobile form fill")
                # Just navigate directly via URL
                page_mobile.goto(f"{BASE_URL}?page=Marketing+Agent", wait_until="networkidle")
                time.sleep(3)
        except Exception as e:
            print(f"Mobile navigation issue: {e}")
        
        # Fill form again
        try:
            inputs_mobile = page_mobile.query_selector_all("input[type='text']")
            if len(inputs_mobile) >= 2:
                inputs_mobile[0].fill("Online Cooking Classes")
                time.sleep(0.5)
                inputs_mobile[1].fill("Busy professionals who want to cook healthy meals")
                time.sleep(0.5)
            
            selects_mobile = page_mobile.query_selector_all("select")
            if len(selects_mobile) >= 2:
                selects_mobile[0].select_option("Sign-ups")
                time.sleep(0.5)
                selects_mobile[1].select_option("Facebook")
                time.sleep(0.5)
            
            # Submit
            submit_mobile = page_mobile.query_selector("button:has-text('Generate Campaign Analysis')")
            if submit_mobile:
                submit_mobile.click()
                time.sleep(6)
        except Exception as e:
            print(f"Note: Mobile form interaction had issues: {e}")
            print("Capturing what we have...")
        
        # Screenshot 3: Mobile result
        print("Capturing mobile result...")
        page_mobile.screenshot(path=f"{SCREENSHOTS_DIR}/marketing_mobile_result.png", full_page=True)
        
        mobile.close()
        
        # Workforce Hub screenshot (desktop)
        hub_context = browser.new_context(viewport={"width": 1280, "height": 800})
        page_hub = hub_context.new_page()
        
        print("Navigating to Workforce Hub...")
        page_hub.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        time.sleep(3)
        
        # Click Workforce Hub
        page_hub.get_by_text("Workforce Hub").click()
        time.sleep(2)
        
        # Screenshot 4: Workforce hub with Marketing card
        print("Capturing workforce hub...")
        page_hub.screenshot(path=f"{SCREENSHOTS_DIR}/marketing_workforce_hub.png", full_page=False)
        
        hub_context.close()
        browser.close()
        
        print("\n✅ All screenshots captured successfully!")
        print(f"Screenshots saved to: {SCREENSHOTS_DIR}")
        print("- marketing_desktop_brief_form.png")
        print("- marketing_desktop_result.png")
        print("- marketing_mobile_result.png")
        print("- marketing_workforce_hub.png")


if __name__ == "__main__":
    capture_marketing_screenshots()
