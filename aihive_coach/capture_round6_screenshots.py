"""
Round 6 Screenshot Capture - Properly Navigate to Each Page
Verifies each screenshot is distinct with md5sum
"""
import os, time, hashlib
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"
os.makedirs(OUT, exist_ok=True)

def md5sum(filepath):
    """Calculate MD5 hash of file"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def wait_for_streamlit(page):
    """Wait for Streamlit to finish rendering"""
    time.sleep(3)
    # Wait for loading indicator to disappear
    page.wait_for_load_state("networkidle")
    time.sleep(2)

screenshots = {}

with sync_playwright() as p:
    browser = p.chromium.launch()
    
    print("=" * 60)
    print("DESKTOP SCREENSHOTS (1920x1080)")
    print("=" * 60)
    
    # Desktop context
    dt_ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
    d = dt_ctx.new_page()
    d.goto(APP_URL)
    wait_for_streamlit(d)
    
    # 1. Homepage Desktop
    print("\n1. Homepage Desktop...")
    d.screenshot(path=f"{OUT}/homepage_desktop.png", full_page=True)
    screenshots["homepage_desktop"] = md5sum(f"{OUT}/homepage_desktop.png")
    print(f"   MD5: {screenshots['homepage_desktop']}")
    
    # 2. Product Cards Section (scroll to products)
    print("\n2. Product Cards Section...")
    d.evaluate("window.scrollTo(0, 800)")
    time.sleep(1)
    d.screenshot(path=f"{OUT}/product_cards_section.png")
    screenshots["product_cards_section"] = md5sum(f"{OUT}/product_cards_section.png")
    print(f"   MD5: {screenshots['product_cards_section']}")
    
    # 3. Founder/Company Section (scroll to bottom)
    print("\n3. Founder/Company Section...")
    d.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(1)
    d.screenshot(path=f"{OUT}/founder_section.png")
    screenshots["founder_section"] = md5sum(f"{OUT}/founder_section.png")
    print(f"   MD5: {screenshots['founder_section']}")
    
    # 4. Navigate to Coach Landing
    print("\n4. Coach Landing Page...")
    d.locator("button:has-text('Launch Coach')").click()
    wait_for_streamlit(d)
    d.screenshot(path=f"{OUT}/coach_landing_desktop.png", full_page=True)
    screenshots["coach_landing_desktop"] = md5sum(f"{OUT}/coach_landing_desktop.png")
    print(f"   MD5: {screenshots['coach_landing_desktop']}")
    
    # 5. Example Journey Section (scroll to it)
    print("\n5. Example Journey Section...")
    d.evaluate("window.scrollTo(0, document.body.scrollHeight - 1500)")
    time.sleep(1)
    d.screenshot(path=f"{OUT}/example_journey_section.png")
    screenshots["example_journey_section"] = md5sum(f"{OUT}/example_journey_section.png")
    print(f"   MD5: {screenshots['example_journey_section']}")
    
    # 6. Navigate to Coaching Workspace
    print("\n6. Coaching Workspace...")
    d.locator("button:has-text('Start Your First Session')").first.click()
    wait_for_streamlit(d)
    # Select Personal Growth category
    try:
        d.locator("button:has-text('Personal Growth')").first.click()
        wait_for_streamlit(d)
    except:
        print("   (Already in coaching workspace)")
    # Just capture the workspace with welcome message
    d.screenshot(path=f"{OUT}/coaching_workspace_desktop.png", full_page=True)
    screenshots["coaching_workspace_desktop"] = md5sum(f"{OUT}/coaching_workspace_desktop.png")
    print(f"   MD5: {screenshots['coaching_workspace_desktop']}")
    
    # 7. Action Plans Page (navigate via sidebar)
    print("\n7. Action Plans Page...")
    # Open sidebar
    try:
        d.locator("[data-testid='stSidebar'] button").first.click()
        time.sleep(1)
    except:
        pass
    d.locator("button:has-text('My Action Plans')").click()
    wait_for_streamlit(d)
    d.screenshot(path=f"{OUT}/action_plans_desktop.png", full_page=True)
    screenshots["action_plans_desktop"] = md5sum(f"{OUT}/action_plans_desktop.png")
    print(f"   MD5: {screenshots['action_plans_desktop']}")
    
    dt_ctx.close()
    
    print("\n" + "=" * 60)
    print("MOBILE SCREENSHOTS (390x844)")
    print("=" * 60)
    
    # Mobile context
    mb_ctx = browser.new_context(viewport={"width": 390, "height": 844})
    m = mb_ctx.new_page()
    m.goto(APP_URL)
    wait_for_streamlit(m)
    
    # 8. Homepage Mobile
    print("\n8. Homepage Mobile...")
    m.screenshot(path=f"{OUT}/homepage_mobile.png", full_page=True)
    screenshots["homepage_mobile"] = md5sum(f"{OUT}/homepage_mobile.png")
    print(f"   MD5: {screenshots['homepage_mobile']}")
    
    # 9. Navigate to Coach Landing Mobile
    print("\n9. Coach Landing Mobile...")
    # Open sidebar on mobile
    try:
        m.locator("[data-testid='stSidebar'] button").first.click()
        time.sleep(1)
    except:
        pass
    m.locator("button:has-text('AIHive Coach')").click()
    wait_for_streamlit(m)
    m.screenshot(path=f"{OUT}/coach_landing_mobile.png", full_page=True)
    screenshots["coach_landing_mobile"] = md5sum(f"{OUT}/coach_landing_mobile.png")
    print(f"   MD5: {screenshots['coach_landing_mobile']}")
    
    # 10. Coaching Workspace Mobile
    print("\n10. Coaching Workspace Mobile...")
    m.locator("button:has-text('Start Your First Session')").first.click()
    wait_for_streamlit(m)
    try:
        m.locator("button:has-text('Personal Growth')").first.click()
        wait_for_streamlit(m)
    except:
        print("   (Already in coaching workspace)")
    m.screenshot(path=f"{OUT}/coaching_workspace_mobile.png", full_page=True)
    screenshots["coaching_workspace_mobile"] = md5sum(f"{OUT}/coaching_workspace_mobile.png")
    print(f"   MD5: {screenshots['coaching_workspace_mobile']}")
    
    # 11. Action Plans Mobile
    print("\n11. Action Plans Mobile...")
    # Open mobile nav expander
    try:
        m.locator("div:has-text('Quick Navigation')").click()
        time.sleep(1)
    except:
        pass
    m.locator("button:has-text('My Plans')").click()
    wait_for_streamlit(m)
    m.screenshot(path=f"{OUT}/action_plans_mobile.png", full_page=True)
    screenshots["action_plans_mobile"] = md5sum(f"{OUT}/action_plans_mobile.png")
    print(f"   MD5: {screenshots['action_plans_mobile']}")
    
    mb_ctx.close()
    browser.close()

print("\n" + "=" * 60)
print("VERIFICATION: ALL MD5 HASHES")
print("=" * 60)
for name, hash in screenshots.items():
    print(f"{name:40s} {hash}")

# Check for duplicates
print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)
hashes_seen = {}
duplicates_found = False
for name, hash in screenshots.items():
    if hash in hashes_seen:
        print(f"❌ DUPLICATE: {name} == {hashes_seen[hash]}")
        duplicates_found = True
    else:
        hashes_seen[hash] = name

if not duplicates_found:
    print("✅ All screenshots are distinct (no duplicates)")
else:
    print("\n⚠️  WARNING: Duplicate screenshots detected!")

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\nAll screenshots saved to: {OUT}")
print(f"Total: {len(screenshots)} files")
