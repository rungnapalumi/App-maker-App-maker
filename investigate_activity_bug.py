"""
Investigate why activity panel is empty
"""
from playwright.sync_api import sync_playwright
import time

def investigate():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        
        print("=== STEP 1: Load hub ===")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(10)
        
        # Check initial activity panel
        body = page.locator("body").inner_text()
        if "No activity yet" in body:
            print("✓ Activity panel shows: 'No activity yet'")
        elif "My Recent Activity" in body:
            print("✓ Activity panel exists")
            # Check if it has entries
            if "AIHive Coach" in body or "AIHive Tutor" in body:
                print("✓ Activity panel has entries!")
            else:
                print("✗ Activity panel is empty")
        
        print("\n=== STEP 2: Open Coach ===")
        # Click Open AIHive Coach
        coach_button = page.locator("button:has-text('Open AIHive Coach')").first
        if coach_button.is_visible():
            print("✓ Found Coach button, clicking...")
            coach_button.click()
            time.sleep(5)
            
            # Check if we're on coach page
            if "AIHive Coach" in page.locator("body").inner_text():
                print("✓ Navigated to Coach page")
            
            # Try to start a coaching session
            # Look for category buttons or start button
            body_text = page.locator("body").inner_text()
            if "Personal Growth" in body_text:
                print("✓ Found category: Personal Growth")
                growth_button = page.locator("button:has-text('Personal Growth')").first
                if growth_button.is_visible():
                    growth_button.click()
                    time.sleep(3)
                    print("✓ Clicked Personal Growth")
        
        print("\n=== STEP 3: Return to hub ===")
        # Click home/hub button in sidebar
        # Need to open sidebar first on some viewports
        try:
            sidebar_toggle = page.locator("button[kind='header']").first
            if sidebar_toggle.count() > 0:
                sidebar_toggle.click()
                time.sleep(1)
        except:
            pass
        
        hub_button = page.locator("button:has-text('Workforce Hub')").first
        if hub_button.is_visible():
            print("✓ Found hub button, clicking...")
            hub_button.click()
            time.sleep(5)
            
            # Check activity panel again
            body = page.locator("body").inner_text()
            if "No activity yet" in body:
                print("✗ Activity panel STILL shows: 'No activity yet'")
                print("   BUG: Session was created but not showing in activity panel")
            elif "Personal Growth" in body or "Improve" in body:
                print("✓ Activity panel now shows session!")
            else:
                print("? Activity panel status unclear")
        
        print("\n=== CHECKING DATABASE ===")
        # Let's check the actual database to see if sessions were created
        import sqlite3
        conn = sqlite3.connect("aihive_coach.db")
        cursor = conn.cursor()
        
        # Check coaching sessions
        cursor.execute("SELECT COUNT(*) FROM coaching_sessions")
        coach_count = cursor.fetchone()[0]
        print(f"Total coaching sessions in DB: {coach_count}")
        
        # Check study sessions  
        cursor.execute("SELECT COUNT(*) FROM study_sessions")
        tutor_count = cursor.fetchone()[0]
        print(f"Total study sessions in DB: {tutor_count}")
        
        # Check unique users
        cursor.execute("SELECT DISTINCT user_id FROM coaching_sessions")
        user_ids = cursor.fetchall()
        print(f"Unique user IDs in coaching_sessions: {len(user_ids)}")
        if len(user_ids) > 0:
            print(f"  First user ID: {user_ids[0][0]}")
        
        conn.close()
        
        browser.close()

if __name__ == "__main__":
    investigate()
