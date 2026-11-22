"""Final screenshot capture - manual verification approach"""
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
    print("Desktop screenshots...")
    d = b.new_context(viewport={"width": 1920, "height": 1080}).new_page()
    d.goto(APP_URL)
    d.wait_for_load_state("networkidle")
    time.sleep(3)
    
    d.screenshot(path=f"{OUT}/homepage_desktop.png", full_page=True)
    print("  homepage_desktop.png")
    screenshots.append(("homepage_desktop.png", md5(f"{OUT}/homepage_desktop.png")))
    
    d.locator("button:has-text('Launch Coach')").click(force=True)
    d.wait_for_load_state("networkidle")
    time.sleep(3)
    d.screenshot(path=f"{OUT}/coach_landing_desktop.png", full_page=True)
    print("  coach_landing_desktop.png")
    screenshots.append(("coach_landing_desktop.png", md5(f"{OUT}/coach_landing_desktop.png")))
    
    d.locator("button:has-text('Start Your First Session')").first.click(force=True)
    d.wait_for_load_state("networkidle")
    time.sleep(3)
    d.screenshot(path=f"{OUT}/coaching_category_select.png", full_page=True)
    print("  coaching_category_select.png")
    screenshots.append(("coaching_category_select.png", md5(f"{OUT}/coaching_category_select.png")))
    
    d.locator("button:has-text('Personal Growth')").first.click(force=True)
    d.wait_for_load_state("networkidle")
    time.sleep(3)
    d.screenshot(path=f"{OUT}/coaching_workspace_desktop.png", full_page=True)
    print("  coaching_workspace_desktop.png")
    screenshots.append(("coaching_workspace_desktop.png", md5(f"{OUT}/coaching_workspace_desktop.png")))
    
    d.close()
    
    # Mobile
    print("\nMobile screenshots...")
    m = b.new_context(viewport={"width": 390, "height": 844}).new_page()
    m.goto(APP_URL)
    m.wait_for_load_state("networkidle")
    time.sleep(3)
    m.screenshot(path=f"{OUT}/homepage_mobile.png", full_page=True)
    print("  homepage_mobile.png")
    screenshots.append(("homepage_mobile.png", md5(f"{OUT}/homepage_mobile.png")))
    
    # Click to coach landing via sidebar
    m.locator("[data-testid='stSidebar']").first.click(force=True, timeout=5000)
    time.sleep(1)
    m.locator("button").filter(has_text="AIHive Coach").click(force=True)
    m.wait_for_load_state("networkidle")
    time.sleep(3)
    m.screenshot(path=f"{OUT}/coach_landing_mobile.png", full_page=True)
    print("  coach_landing_mobile.png")
    screenshots.append(("coach_landing_mobile.png", md5(f"{OUT}/coach_landing_mobile.png")))
    
    m.locator("button").filter(has_text="Start Your First Session").first.click(force=True)
    m.wait_for_load_state("networkidle")
    time.sleep(3)
    m.locator("button").filter(has_text="Personal Growth").first.click(force=True)
    m.wait_for_load_state("networkidle")
    time.sleep(3)
    m.screenshot(path=f"{OUT}/coaching_workspace_mobile.png", full_page=True)
    print("  coaching_workspace_mobile.png")
    screenshots.append(("coaching_workspace_mobile.png", md5(f"{OUT}/coaching_workspace_mobile.png")))
    
    m.close()
    b.close()

print("\n" + "="*60)
print("MD5 HASHES")
print("="*60)
for name, hash in screenshots:
    print(f"{name:40s} {hash}")

hashes = [h for _, h in screenshots]
if len(hashes) == len(set(hashes)):
    print("\n✅ All distinct")
else:
    print("\n❌ Duplicates:")
    seen = {}
    for name, hash in screenshots:
        if hash in seen:
            print(f"  {name} == {seen[hash]}")
        seen[hash] = name
