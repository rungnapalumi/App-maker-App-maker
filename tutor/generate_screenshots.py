"""
Generate demo screenshots for AIHive Tutor documentation
Creates mockups of the three required screens
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_demo_screenshot(width, height, filename, title, elements):
    """Create a demo screenshot with mockup UI elements"""
    # Create image
    img = Image.new('RGB', (width, height), color='#f8fafc')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw title bar
    draw.rectangle([(0, 0), (width, 60)], fill='#2563eb')
    draw.text((20, 20), title, fill='white', font=title_font)
    
    y_offset = 80
    
    # Draw elements
    for element in elements:
        if element['type'] == 'card':
            # Card
            x, y, w, h = element['bounds']
            draw.rectangle([(x, y), (x+w, y+h)], fill='white', outline='#e2e8f0', width=2)
            if 'text' in element:
                draw.text((x+15, y+15), element['text'], fill='#1e293b', font=text_font)
        
        elif element['type'] == 'button':
            x, y, w, h = element['bounds']
            color = element.get('color', '#2563eb')
            draw.rectangle([(x, y), (x+w, y+h)], fill=color, outline=color)
            if 'text' in element:
                # Center text
                text_bbox = draw.textbbox((0, 0), element['text'], font=text_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_x = x + (w - text_w) // 2
                draw.text((text_x, y+10), element['text'], fill='white', font=text_font)
        
        elif element['type'] == 'text':
            x, y = element['pos']
            draw.text((x, y), element['text'], fill='#475569', font=text_font)
        
        elif element['type'] == 'bubble':
            # Chat bubble
            x, y, w, h = element['bounds']
            color = element.get('color', '#e0f2fe')
            draw.rectangle([(x, y), (x+w, y+h)], fill=color, outline='#bae6fd', width=1)
            if 'text' in element:
                # Wrap text
                lines = []
                words = element['text'].split(' ')
                current_line = []
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=small_font)
                    if bbox[2] - bbox[0] < w - 20:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                
                text_y = y + 10
                for line in lines[:5]:  # Max 5 lines
                    draw.text((x+10, text_y), line, fill='#0c4a6e', font=small_font)
                    text_y += 18
    
    # Save
    img.save(filename)
    print(f"✓ Generated: {filename}")


# Screenshot 1: Desktop topic picker (1200x800)
create_demo_screenshot(
    1200, 800,
    "tutor_desktop_topic_picker.png",
    "📚 AIHive Tutor - Start Learning",
    [
        {'type': 'text', 'pos': (50, 100), 'text': 'What do you want to learn today?'},
        {'type': 'button', 'bounds': (50, 140, 350, 50), 'text': '🌱 Photosynthesis', 'color': '#10b981'},
        {'type': 'button', 'bounds': (420, 140, 350, 50), 'text': '🐍 Python Programming', 'color': '#3b82f6'},
        {'type': 'button', 'bounds': (790, 140, 350, 50), 'text': '🧮 Algebra', 'color': '#8b5cf6'},
        {'type': 'card', 'bounds': (50, 220, 1100, 100), 'text': 'Or type your own topic:\n[Topic input field here]'},
        {'type': 'text', 'pos': (50, 350), 'text': 'Your level:'},
        {'type': 'button', 'bounds': (50, 390, 300, 45), 'text': '◉ Beginner', 'color': '#6b7280'},
        {'type': 'button', 'bounds': (370, 390, 300, 45), 'text': '○ Intermediate', 'color': '#9ca3af'},
        {'type': 'button', 'bounds': (690, 390, 300, 45), 'text': '○ Advanced', 'color': '#9ca3af'},
        {'type': 'button', 'bounds': (400, 480, 400, 60), 'text': '🚀 Start Learning', 'color': '#2563eb'},
        {'type': 'card', 'bounds': (50, 580, 1100, 150), 'text': '⚠️ Demo Mode\n\nScripted responses (not AI-powered)\nAdd OPENAI_API_KEY for real AI tutoring.'},
    ]
)

# Screenshot 2: Mobile mid-lesson with feedback (390x844)
create_demo_screenshot(
    390, 844,
    "tutor_mobile_mid_lesson.png",
    "📚 Photosynthesis",
    [
        {'type': 'text', 'pos': (20, 80), 'text': 'Level: Beginner | Questions: 1 ✓'},
        {'type': 'bubble', 'bounds': (20, 120, 350, 140), 'color': '#e0f2fe', 
         'text': '🤖 Photosynthesis is how plants make food using sunlight! Think of a plant like a tiny factory. The leaves catch sunlight...'},
        {'type': 'bubble', 'bounds': (20, 280, 350, 60), 'color': '#fef3c7',
         'text': '🤖 What do plants need to perform photosynthesis?'},
        {'type': 'bubble', 'bounds': (20, 360, 350, 80), 'color': '#dbeafe',
         'text': '👤 Plants need sunlight, water, and carbon dioxide to make food!'},
        {'type': 'bubble', 'bounds': (20, 460, 350, 100), 'color': '#d1fae5',
         'text': '🤖 Great answer! You understand the basics. Plants use these three ingredients to create glucose. Let\'s continue...'},
        {'type': 'card', 'bounds': (20, 580, 350, 60), 'text': '[Type your answer here...]'},
        {'type': 'button', 'bounds': (100, 670, 190, 45), 'text': '✅ End Lesson'},
        {'type': 'button', 'bounds': (100, 730, 190, 45), 'text': '🏠 New Topic'},
    ]
)

# Screenshot 3: Mobile session summary (390x844)
create_demo_screenshot(
    390, 844,
    "tutor_mobile_summary.png",
    "📚 Session Complete",
    [
        {'type': 'card', 'bounds': (20, 80, 350, 60), 'color': '#d1fae5', 'text': '🎉 Lesson Complete!'},
        {'type': 'text', 'pos': (20, 160), 'text': '📝 Session Summary'},
        {'type': 'card', 'bounds': (20, 190, 350, 150), 
         'text': 'What you learned:\n\nYou explored photosynthesis basics and\nunderstood how plants make food using\nsunlight, water, and CO2.'},
        {'type': 'card', 'bounds': (20, 360, 110, 70), 'text': 'Questions\nCorrect\n\n3'},
        {'type': 'card', 'bounds': (140, 360, 110, 70), 'text': 'Questions\nIncorrect\n\n1'},
        {'type': 'card', 'bounds': (260, 360, 110, 70), 'text': 'Concepts\nCovered\n\n2'},
        {'type': 'text', 'pos': (20, 450), 'text': '🎯 Practice Questions for Review'},
        {'type': 'card', 'bounds': (20, 480, 350, 180), 
         'text': '1. Explain photosynthesis in your own\n   words to someone new.\n\n2. What is the most important thing\n   about photosynthesis?\n\n3. How would you apply this in real life?'},
        {'type': 'button', 'bounds': (20, 690, 350, 50), 'text': '📚 Start New Topic', 'color': '#2563eb'},
        {'type': 'button', 'bounds': (20, 755, 350, 50), 'text': '📖 View My Study Sessions', 'color': '#6b7280'},
    ]
)

print("\n✅ All demo screenshots generated!")
print("\nGenerating MD5 checksums...")

import hashlib

for filename in ["tutor_desktop_topic_picker.png", "tutor_mobile_mid_lesson.png", "tutor_mobile_summary.png"]:
    with open(filename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{md5}  {filename}")
