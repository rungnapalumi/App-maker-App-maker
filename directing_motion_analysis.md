# Directing Motion Analysis

## 🎯 **Motion Definition**

**Directing** is a fundamental Laban Movement Analysis (LMA) motion type characterized by:

### **Key Characteristics:**
- **Direction**: Straight toward target
- **Body Part Involvement**: Pointing/reaching with one joint chain
- **Pathway**: Linear and focused
- **Timing**: Sustained or quick
- **Other Motion Clues**: Eyes and head align with hand/target

### **LMA Context:**
Directing represents **intentional, focused movement** where the body extends toward a specific target or direction. It's often used in:
- Pointing gestures
- Reaching movements
- Directional communication
- Focused attention movements

## 🔍 **Detection Algorithm**

### **Technical Implementation:**
The Directing motion is detected using the following criteria:

1. **Arm Extension Detection:**
   ```python
   left_arm_extended = (left_elbow_angle > 150 and lw[2] < ls[2] - 0.1)
   right_arm_extended = (right_elbow_angle > 150 and rw[2] < rs[2] - 0.1)
   ```

2. **Asymmetric Pointing:**
   - One arm must be extended forward
   - The other arm should NOT be extended (creating asymmetry)

3. **Movement Magnitude:**
   ```python
   pointing_magnitude = np.linalg.norm(pointing_arm_delta[:2])
   if pointing_magnitude > 0.01 and pointing_magnitude < 0.05:
   ```

### **Detection Parameters:**
- **Elbow Angle**: > 150° (nearly straight arm)
- **Forward Extension**: Wrist must be in front of shoulder (Z-axis)
- **Movement Range**: 0.01 to 0.05 units (moderate, focused movement)
- **Asymmetry**: Only one arm extended at a time

## 🎬 **How to Perform Directing Motion**

### **Basic Directing Gesture:**
1. **Stand or sit** in a neutral position
2. **Extend one arm** straight forward (elbow nearly straight)
3. **Keep the other arm** relaxed at your side
4. **Point toward a specific target** or direction
5. **Hold the position** for 1-2 seconds
6. **Maintain focus** with your eyes aligned with your pointing hand

### **Variations to Test:**
1. **Left Arm Directing**: Extend left arm forward, keep right arm down
2. **Right Arm Directing**: Extend right arm forward, keep left arm down
3. **Upward Directing**: Point upward with one arm
4. **Side Directing**: Point to the side with one arm
5. **Downward Directing**: Point downward with one arm

### **Common Mistakes to Avoid:**
- ❌ **Both arms extended** (this would be "Spreading")
- ❌ **Quick flicking motion** (this would be "Flicking")
- ❌ **Arms close together** (this would be "Enclosing")
- ❌ **Forceful pushing motion** (this would be "Punching")

## 📊 **Detection Accuracy**

### **Expected Detection:**
- ✅ **Single arm pointing** forward
- ✅ **Sustained pointing** gesture
- ✅ **Focused directional** movement
- ✅ **Asymmetric arm** positioning

### **May Not Detect:**
- ❌ **Very subtle** pointing movements
- ❌ **Extremely fast** pointing gestures
- ❌ **Both arms** pointing simultaneously
- ❌ **Pointing with bent** elbows

## 🔧 **Technical Details**

### **Landmark Analysis:**
- **Shoulder Position**: Reference point for extension
- **Elbow Angle**: Measures arm straightness
- **Wrist Position**: Determines pointing direction
- **Movement Delta**: Calculates motion magnitude

### **Coordinate System:**
- **X-axis**: Left-right movement
- **Y-axis**: Up-down movement  
- **Z-axis**: Forward-backward movement (depth)

### **Threshold Values:**
- **Elbow Angle Threshold**: 150°
- **Forward Extension**: -0.1 units (wrist in front of shoulder)
- **Movement Range**: 0.01 - 0.05 units
- **Asymmetry Check**: Only one arm extended

## 🎯 **Testing Recommendations**

### **Best Practices:**
1. **Clear pointing** with straight arm
2. **Hold position** for 2-3 seconds
3. **Use one arm** at a time
4. **Point to specific** objects or directions
5. **Maintain steady** posture

### **Video Recording Tips:**
- **Good lighting** for clear landmark detection
- **Full body** in frame
- **Side angle** to show arm extension
- **Steady camera** position
- **Multiple takes** with different directions

## 📈 **Performance Metrics**

### **Detection Rate:**
- **High Accuracy**: Clear, sustained pointing gestures
- **Medium Accuracy**: Quick pointing movements
- **Low Accuracy**: Subtle or ambiguous gestures

### **False Positives:**
- May detect "Pressing" if movement is too forceful
- May detect "Spreading" if both arms are involved
- May detect "Punching" if movement is too sudden

### **False Negatives:**
- Very subtle pointing movements
- Pointing with bent elbows
- Rapid pointing gestures
- Pointing while moving body significantly

---

*This analysis is part of the LMA Motion Analysis System v2.7*
*For technical support, refer to the main system documentation.* 