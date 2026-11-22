"""
Capture remaining screenshots
"""
import os
import time
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUTPUT_DIR = "/opt/cursor/artifacts"


def capture_remaining():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        
        # Mobile coaching workspace
        print("Capturing mobile coaching workspace...")
        mobile = browser.new_context(viewport={"width": 375, "height": 812})
        mobile_page = mobile.new_page()
        
        mobile_page.goto(APP_URL)
        time.sleep(3)
        
        # Navigate to coaching via sidebar
        mobile_page.locator("button[kind='secondary']").first.click()  # Start Coaching button in content
        time.sleep(2)
        
        # Click category button
        mobile_page.locator("button:has-text('Personal Growth')").click()
        time.sleep(3)
        
        mobile_page.screenshot(path=f"{OUTPUT_DIR}/3_coaching_workspace_mobile.png", full_page=True)
        
        # Desktop action plans
        print("Capturing desktop action plans...")
        desktop = browser.new_context(viewport={"width": 1920, "height": 1080})
        desktop_page = desktop.new_page()
        
        desktop_page.goto(APP_URL)
        time.sleep(3)
        
        # Click My Action Plans in sidebar
        desktop_page.locator("text=My Action Plans").click()
        time.sleep(2)
        
        desktop_page.screenshot(path=f"{OUTPUT_DIR}/4_action_plans_desktop.png", full_page=True)
        
        # Mobile action plans
        print("Capturing mobile action plans...")
        mobile_page.locator("text=My Action Plans").click()
        time.sleep(2)
        
        mobile_page.screenshot(path=f"{OUTPUT_DIR}/4_action_plans_mobile.png", full_page=True)
        
        print("Done!")
        
        mobile.close()
        desktop.close()
        browser.close()


if __name__ == "__main__":
    capture_remaining()
