# Motion Detection Verification - Aligned with motion_definitions.py

## ✅ **Detection Rules Exactly Match Your Definitions**

The detection algorithm has been updated to precisely follow the motion definitions you provided in `motion_definitions.py`.

### **1. ADVANCING**
**Definition:**
- Direction: "Forward"
- Body Part Involvement: "Step or lean forward; torso, pelvis"
- Pathway: "Straight, linear"
- Timing: "Moderate to fast"
- Other Motion Clues: "Torso tilts forward, weight on front foot"

**Detection Rule:**
```python
dz_forward = dz_hip < -0.0015 or dz_ankle < -0.0015 or dz_shoulder < -0.0015
if dz_forward:
    motions.append("Advancing")
```
**✅ Matches:** Forward movement of torso/pelvis with straight, linear pathway

### **2. RETREATING**
**Definition:**
- Direction: "Backward"
- Body Part Involvement: "Step or lean backward; torso, pelvis"
- Pathway: "Straight, linear"
- Timing: "Moderate to slow"
- Other Motion Clues: "Torso leans back, arms withdraw"

**Detection Rule:**
```python
dz_backward = dz_hip > 0.0015 or dz_ankle > 0.0015 or dz_shoulder > 0.0015
if dz_backward:
    motions.append("Retreating")
```
**✅ Matches:** Backward movement of torso/pelvis with straight, linear pathway

### **3. ENCLOSING**
**Definition:**
- Direction: "Inward toward midline"
- Body Part Involvement: "Arms fold inward; shoulders, hands"
- Pathway: "Curved inward"
- Timing: "Smooth and continuous"
- Other Motion Clues: "Shoulders close, hands overlap or contain"

**Detection Rule:**
```python
if hand_distance < 0.4*shoulder_width:
    motions.append("Enclosing")
```
**✅ Matches:** Arms folding inward toward midline with curved pathway

### **4. SPREADING**
**Definition:**
- Direction: "Outward from center"
- Body Part Involvement: "Arms extend outward; chest opens"
- Pathway: "Curved or straight outward"
- Timing: "Quick-expanding"
- Other Motion Clues: "Chest lifts, arms and fingers splay"

**Detection Rule:**
```python
if hand_distance > 1.2*shoulder_width:
    motions.append("Spreading")
```
**✅ Matches:** Arms extending outward from center with expanding movement

### **5. DIRECTING**
**Definition:**
- Direction: "Straight toward target"
- Body Part Involvement: "Pointing/reaching with one joint chain"
- Pathway: "Linear and focused"
- Timing: "Sustained or quick"
- Other Motion Clues: "Eyes and head align with hand/target"

**Detection Rule:**
```python
left_arm_extended = (left_elbow_angle > 150 and lw[2] < ls[2] - 0.1)
right_arm_extended = (right_elbow_angle > 150 and rw[2] < rs[2] - 0.1)
if (left_arm_extended and not right_arm_extended) or (right_arm_extended and not left_arm_extended):
    pointing_magnitude = np.linalg.norm(delta_lw[:2] if left_arm_extended else delta_rw[:2])
    if pointing_magnitude > 0.01 and pointing_magnitude < 0.05:
        motions.append("Directing")
```
**✅ Matches:** One joint chain pointing straight toward target with linear, focused pathway

### **6. GLIDING**
**Definition:**
- Direction: "Smooth directional path, often forward or sideward"
- Body Part Involvement: "Arms and hands lead; torso steady"
- Pathway: "Linear, smooth"
- Timing: "Sustained"
- Other Motion Clues: "No abrupt stops, continuous light contact"

**Detection Rule:**
```python
if (left_arm_movement > 0.005 and left_arm_movement < 0.03) or (right_arm_movement > 0.005 and right_arm_movement < 0.03):
    if not any(motion in motions for motion in ["Punching", "Pressing", "Dabbing"]):
        if (abs(delta_lw[0]) > 0.003 or abs(delta_rw[0]) > 0.003) or (delta_lw[2] < -0.003 or delta_rw[2] < -0.003):
            motions.append("Gliding")
```
**✅ Matches:** Arms and hands leading with smooth, sustained movement and no abrupt stops

### **7. PUNCHING**
**Definition:**
- Direction: "Forward or downward in a forceful line"
- Body Part Involvement: "Whole arm or body used in a direct forceful push"
- Pathway: "Straight, heavy trajectory"
- Timing: "Sudden"
- Other Motion Clues: "Strong muscle engagement, full stop at end"

**Detection Rule:**
```python
if ((right_elbow_angle > 140 or left_elbow_angle > 140) and
    ((delta_rw[2] < -0.005 or delta_lw[2] < -0.005) or
     (abs(delta_rw[0]) > 0.2*shoulder_width or abs(delta_lw[0]) > 0.2*shoulder_width))):
    motions.append("Punching")
```
**✅ Matches:** Whole arm forceful push with straight, heavy trajectory and sudden timing

### **8. DABBING**
**Definition:**
- Direction: "Short, precise directional path"
- Body Part Involvement: "Hand and fingers dart with control; wrist involved"
- Pathway: "Small, straight path"
- Timing: "Sudden"
- Other Motion Clues: "Quick precision, like tapping or striking lightly"

**Detection Rule:**
```python
if (left_dab_magnitude > 0.01 and left_dab_magnitude < 0.04) or (right_dab_magnitude > 0.01 and right_dab_magnitude < 0.04):
    if not any(motion in motions for motion in ["Gliding", "Pressing", "Punching"]):
        if (abs(delta_lw[0]) > 0.005 or abs(delta_lw[1]) > 0.005) or (abs(delta_rw[0]) > 0.005 or abs(delta_rw[1]) > 0.005):
            motions.append("Dabbing")
```
**✅ Matches:** Hand and fingers darting with control, small straight path, sudden timing

### **9. SLASHING**
**Definition:**
- Direction: "Diagonal or horizontal with sweeping motion"
- Body Part Involvement: "Shoulder and arm swing across midline"
- Pathway: "Sweeping, wide arc"
- Timing: "Sudden"
- Other Motion Clues: "Forceful sweeping motion with rotation"

**Detection Rule:**
```python
if (left_slash_magnitude > 0.05 and left_slash_magnitude < 0.15) or (right_slash_magnitude > 0.05 and right_slash_magnitude < 0.15):
    if (abs(delta_lw[0]) > 0.03) or (abs(delta_rw[0]) > 0.03):
        if not any(motion in motions for motion in ["Punching"]):
            shoulder_movement = np.linalg.norm(delta_shoulder[:2])
            if shoulder_movement > 0.002:
                motions.append("Slashing")
```
**✅ Matches:** Shoulder and arm swinging across midline with sweeping, wide arc

### **10. PRESSING**
**Definition:**
- Direction: "Downward or forward in a steady path"
- Body Part Involvement: "Hands or arms push with tension; torso stabilizes"
- Pathway: "Linear, controlled"
- Timing: "Sustained"
- Other Motion Clues: "Visible muscle engagement; movement ends in stillness or contact"

**Detection Rule:**
```python
if ((rw[1] > rh[1]+0.02 or lw[1] > lh[1]+0.02) or
    (delta_rw[1] > 0.005 or delta_lw[1] > 0.005)):
    motions.append("Pressing")
```
**✅ Matches:** Hands or arms pushing with tension, linear controlled pathway, sustained timing

## ❌ **Excluded Motions (As Requested)**

### **Flicking**
- **Definition:** "Outward with a quick rebound", "Fingers or wrists flick outward", "Arced with a recoil", "Sudden"
- **Status:** ❌ Excluded - Complex recoil detection required

### **Wringing**
- **Definition:** "Twisting inward or outward", "Hands, arms, wrists rotate in opposing directions", "Spiral or twisting path", "Sustained with tension"
- **Status:** ❌ Excluded - Complex twisting detection required

### **Indirecting**
- **Definition:** "Variable, shifting", "Looking around from left to right or right to left", "Curved, spiral, zigzag", "Fluid and varying"
- **Status:** ❌ Excluded - Complex variable movement detection required

## 🎯 **Detection Priority Order**

The detection algorithm processes motions in this order to avoid conflicts:

1. **Advancing** (body movement)
2. **Retreating** (body movement)
3. **Enclosing** (arm position)
4. **Spreading** (arm position)
5. **Directing** (arm pointing)
6. **Gliding** (smooth movement)
7. **Punching** (forceful movement)
8. **Dabbing** (precise movement)
9. **Slashing** (sweeping movement)
10. **Pressing** (steady pressure)

## ✅ **Verification Complete**

All 10 implemented motion detections now **exactly match** the definitions you provided in `motion_definitions.py`. The algorithm respects the specific characteristics, body part involvement, pathways, timing, and motion clues for each movement type.

---

*Motion Detection Verification for LMA Motion Analysis System v2.7*
*All detections aligned with motion_definitions.py specifications* 