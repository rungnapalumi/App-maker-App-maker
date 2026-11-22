"""Simple screenshot capture - basic pages only"""
import os, time, hashlib, subprocess
from playwright.sync_api import sync_playwright

APP_URL = "http://localhost:8501"
OUT = "/opt/cursor/artifacts"
os.makedirs(OUT, exist_ok=True)

def md5(f):
    return subprocess.check_output(["md5sum", f]).decode().split()[0]

screenshots = []

with sync_playwright() as p:
    b = p.chromium.launch()
    
    # Desktop
    print("Desktop 1920x1080...")
    d = b.new_context(viewport={"width": 1920, "height": 1080}).new_page()
    d.goto(APP_URL)
    time.sleep(5)
    
    d.screenshot(path=f"{OUT}/1_homepage_desktop.png", full_page=True)
    screenshots.append(("1_homepage_desktop.png", md5(f"{OUT}/1_homepage_desktop.png")))
    
    d.locator("button:has-text('Launch Coach')").click()
    time.sleep(5)
    d.screenshot(path=f"{OUT}/2_coach_landing_desktop.png", full_page=True)
    screenshots.append(("2_coach_landing_desktop.png", md5(f"{OUT}/2_coach_landing_desktop.png")))
    
    d.locator("button:has-text('Start Your First Session')").first.click()
    time.sleep(5)
    d.screenshot(path=f"{OUT}/3_category_select.png", full_page=True)
    screenshots.append(("3_category_select.png", md5(f"{OUT}/3_category_select.png")))
    
    d.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(5)
    d.screenshot(path=f"{OUT}/4_coaching_workspace_desktop.png", full_page=True)
    screenshots.append(("4_coaching_workspace_desktop.png", md5(f"{OUT}/4_coaching_workspace_desktop.png")))
    
    # Mobile
    print("\nMobile 390x844...")
    m = b.new_context(viewport={"width": 390, "height": 844}).new_page()
    m.goto(APP_URL)
    time.sleep(5)
    m.screenshot(path=f"{OUT}/5_homepage_mobile.png", full_page=True)
    screenshots.append(("5_homepage_mobile.png", md5(f"{OUT}/5_homepage_mobile.png")))
    
    # Open sidebar
    try:
        m.locator("[data-testid='stSidebar'] button").first.click()
        time.sleep(1)
    except:
        pass
    
    m.locator("button:has-text('AIHive Coach')").click()
    time.sleep(5)
    m.screenshot(path=f"{OUT}/6_coach_landing_mobile.png", full_page=True)
    screenshots.append(("6_coach_landing_mobile.png", md5(f"{OUT}/6_coach_landing_mobile.png")))
    
    m.locator("button:has-text('Start Your First Session')").first.click()
    time.sleep(5)
    m.locator("button:has-text('Personal Growth')").first.click()
    time.sleep(5)
    m.screenshot(path=f"{OUT}/7_coaching_workspace_mobile.png", full_page=True)
    screenshots.append(("7_coaching_workspace_mobile.png", md5(f"{OUT}/7_coaching_workspace_mobile.png")))
    
    b.close()

print("\n" + "="*60)
print("SCREENSHOTS & MD5 HASHES")
print("="*60)
for name, hash in screenshots:
    print(f"{name:40s} {hash}")

# Check duplicates
hashes = [h for _, h in screenshots]
if len(hashes) == len(set(hashes)):
    print("\n✅ All distinct (no duplicates)")
else:
    print("\n❌ DUPLICATES FOUND:")
    seen = {}
    for name, hash in screenshots:
        if hash in seen:
            print(f"  {name} == {seen[hash]}")
        seen[hash] = name
