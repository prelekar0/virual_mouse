#!/usr/bin/env python3
"""
Virtual Mouse Examples

This script demonstrates how to use the virtual mouse with different settings
and explains the gestures with visual examples.
"""

import cv2
import numpy as np
import os
import argparse

def create_gesture_example(name, description, image_size=(300, 200)):
    """Create an example image for a gesture"""
    # Create a blank image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Add gesture name as title
    cv2.putText(
        image,
        name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )
    
    # Add gesture description
    # Split description into lines that fit on the image
    words = description.split()
    lines = []
    line = ""
    
    for word in words:
        if len(line + word) < 35:  # approximate max chars per line
            line += word + " "
        else:
            lines.append(line)
            line = word + " "
    
    if line:
        lines.append(line)
    
    # Draw each line
    for i, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 70 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
    
    return image

def show_examples():
    """Show examples of gestures and controls"""
    # Create example images
    examples = [
        ("Cursor Movement", "Move your hand naturally to control the cursor position."),
        ("Left Click (Fist)", "Make a fist by closing all your fingers to perform a left click."),
        ("Right Click (Two Fingers)", "Extend only your index and middle fingers to perform a right click."),
        ("Scroll Mode (Open Palm)", "Extend all fingers in an open palm gesture, then move up/down or left/right to scroll.")
    ]
    
    # Create and show each example
    for name, description in examples:
        example = create_gesture_example(name, description)
        cv2.imshow(name, example)
    
    print("Press any key on any window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_usage():
    """Display usage information"""
    print("\nVirtual Mouse Usage Examples")
    print("==========================\n")
    
    print("Basic Usage:")
    print("  python virtual_mouse.py\n")
    
    print("Advanced Options:")
    print("  # Use a different camera (e.g., external webcam)")
    print("  python virtual_mouse.py --camera 1\n")
    
    print("  # Adjust smoothing factor (0.0-1.0, higher = smoother but more lag)")
    print("  python virtual_mouse.py --smoothing 0.7\n")
    
    print("  # Change click cooldown time in seconds")
    print("  python virtual_mouse.py --cooldown 0.3\n")
    
    print("  # Show hand landmarks")
    print("  python virtual_mouse.py --show-landmarks\n")
    
    print("  # Set camera resolution")
    print("  python virtual_mouse.py --width 800 --height 600\n")
    
    print("Full Command with All Options:")
    print("  python virtual_mouse.py --camera 0 --smoothing 0.5 --cooldown 0.5 --show-landmarks --width 640 --height 480\n")
    
    print("Testing Environment:")
    print("  # Check if all required dependencies are installed")
    print("  python test_environment.py\n")

def main():
    parser = argparse.ArgumentParser(description="Virtual Mouse Examples")
    parser.add_argument("--visual", action="store_true", help="Show visual examples of gestures")
    args = parser.parse_args()
    
    if args.visual:
        show_examples()
    
    show_usage()
    
    print("Ready to try the virtual mouse? Run:")
    print("  python virtual_mouse.py")

if __name__ == "__main__":
    main() 