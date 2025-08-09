"""
Motion Type Definitions
=======================

This file contains the definitions for different motion types based on their characteristics:
- Direction
- Body Part Involvement  
- Pathway
- Timing
- Other Motion Clues

These definitions can be used for motion analysis and classification in the skeleton overlay system.
"""

MOTION_DEFINITIONS = {
    "Advancing": {
        "direction": "Forward",
        "body_part_involvement": "Step or lean forward; torso, pelvis",
        "pathway": "Straight, linear",
        "timing": "Moderate to fast",
        "other_motion_clues": "Torso tilts forward, weight on front foot"
    },
    "Retreating": {
        "direction": "Backward", 
        "body_part_involvement": "Step or lean backward; torso, pelvis",
        "pathway": "Straight, linear",
        "timing": "Moderate to slow",
        "other_motion_clues": "Torso leans back, arms withdraw"
    },
    "Enclosing": {
        "direction": "Inward toward midline",
        "body_part_involvement": "Arms fold inward; shoulders, hands",
        "pathway": "Curved inward",
        "timing": "Smooth and continuous",
        "other_motion_clues": "Shoulders close, hands overlap or contain"
    },
    "Spreading": {
        "direction": "Outward from center",
        "body_part_involvement": "Arms extend outward; chest opens",
        "pathway": "Curved or straight outward",
        "timing": "Quick-expanding",
        "other_motion_clues": "Chest lifts, arms and fingers splay"
    },
    "Directing": {
        "direction": "Straight toward target",
        "body_part_involvement": "Pointing/reaching with one joint chain",
        "pathway": "Linear and focused",
        "timing": "Sustained or quick",
        "other_motion_clues": "Eyes and head align with hand/target"
    },
    "Indirecting": {
        "direction": "Variable, shifting",
        "body_part_involvement": "Looking around from left to right or right to left",
        "pathway": "Curved, spiral, zigzag",
        "timing": "Fluid and varying",
        "other_motion_clues": "Eyes scan, hands scoop or weave"
    },
    "Gliding": {
        "direction": "Smooth directional path, often forward or sideward",
        "body_part_involvement": "Arms and hands lead; torso steady",
        "pathway": "Linear, smooth",
        "timing": "Sustained",
        "other_motion_clues": "No abrupt stops, continuous light contact"
    },
    "Punching": {
        "direction": "Forward or downward in a forceful line",
        "body_part_involvement": "Whole arm or body used in a direct forceful push",
        "pathway": "Straight, heavy trajectory",
        "timing": "Sudden",
        "other_motion_clues": "Strong muscle engagement, full stop at end"
    },
    "Dabbing": {
        "direction": "Short, precise directional path",
        "body_part_involvement": "Hand and fingers dart with control; wrist involved",
        "pathway": "Small, straight path",
        "timing": "Sudden",
        "other_motion_clues": "Quick precision, like tapping or striking lightly"
    },
    "Flicking": {
        "direction": "Outward with a quick rebound",
        "body_part_involvement": "Fingers or wrists flick outward; elbow may extend",
        "pathway": "Arced with a recoil",
        "timing": "Sudden",
        "other_motion_clues": "Ends with a recoil or rebound"
    },
    "Slashing": {
        "direction": "Diagonal or horizontal with sweeping motion",
        "body_part_involvement": "Shoulder and arm swing across midline",
        "pathway": "Sweeping, wide arc",
        "timing": "Sudden",
        "other_motion_clues": "Forceful sweeping motion with rotation"
    },
    "Wringing": {
        "direction": "Twisting inward or outward",
        "body_part_involvement": "Hands, arms, wrists rotate in opposing directions",
        "pathway": "Spiral or twisting path",
        "timing": "Sustained with tension",
        "other_motion_clues": "Visible effort in spiraling or twisting hands/arms"
    },
    "Pressing": {
        "direction": "Downward or forward in a steady path",
        "body_part_involvement": "Hands or arms push with tension; torso stabilizes",
        "pathway": "Linear, controlled",
        "timing": "Sustained",
        "other_motion_clues": "Visible muscle engagement; movement ends in stillness or contact"
    }
}

def get_motion_definition(motion_type):
    """
    Get the definition for a specific motion type.
    
    Args:
        motion_type (str): The name of the motion type
        
    Returns:
        dict: The motion definition dictionary, or None if not found
    """
    return MOTION_DEFINITIONS.get(motion_type)

def get_all_motion_types():
    """
    Get a list of all available motion types.
    
    Returns:
        list: List of motion type names
    """
    return list(MOTION_DEFINITIONS.keys())

def get_motion_characteristics(motion_type, characteristic):
    """
    Get a specific characteristic of a motion type.
    
    Args:
        motion_type (str): The name of the motion type
        characteristic (str): The characteristic to retrieve (direction, body_part_involvement, 
                            pathway, timing, other_motion_clues)
        
    Returns:
        str: The characteristic value, or None if not found
    """
    motion_def = get_motion_definition(motion_type)
    if motion_def:
        return motion_def.get(characteristic)
    return None

def print_motion_definitions():
    """
    Print all motion definitions in a formatted way.
    """
    print("Motion Type Definitions")
    print("=" * 50)
    
    for motion_type, definition in MOTION_DEFINITIONS.items():
        print(f"\n{motion_type.upper()}")
        print("-" * len(motion_type))
        for key, value in definition.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    # Example usage
    print_motion_definitions() 