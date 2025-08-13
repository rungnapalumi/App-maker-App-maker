#!/usr/bin/env python3
"""
Video Source Checker
This script checks where the videos are being sourced from in the app.
"""

import os
from pathlib import Path

def check_video_sources():
    print("🔍 Checking Video Sources...")
    print("=" * 50)
    
    # Check local video files
    video1_path = Path("Movement matters.mp4")
    video2_path = Path("The key to effective public speaking  your body movement.mp4")
    
    print(f"📁 Video 1 Local Path: {video1_path.absolute()}")
    print(f"   Exists: {video1_path.exists()}")
    if video1_path.exists():
        print(f"   Size: {video1_path.stat().st_size / (1024*1024):.1f} MB")
    
    print(f"📁 Video 2 Local Path: {video2_path.absolute()}")
    print(f"   Exists: {video2_path.exists()}")
    if video2_path.exists():
        print(f"   Size: {video2_path.stat().st_size / (1024*1024):.1f} MB")
    
    print("\n🌐 Environment Variables:")
    video1_url = os.getenv("VIDEO1_URL", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    video2_url = os.getenv("VIDEO2_URL", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    
    print(f"   VIDEO1_URL: {video1_url}")
    print(f"   VIDEO2_URL: {video2_url}")
    
    print("\n🎯 Source Priority Logic:")
    print("   1. Check if local file exists")
    print("   2. If not, use remote URL")
    print("   3. If neither, show fallback")
    
    print("\n📊 Current Source Status:")
    if video1_path.exists():
        print("   ✅ Video 1: LOCAL FILE")
    else:
        print("   🌐 Video 1: REMOTE URL")
    
    if video2_path.exists():
        print("   ✅ Video 2: LOCAL FILE")
    else:
        print("   🌐 Video 2: REMOTE URL")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_video_sources() 