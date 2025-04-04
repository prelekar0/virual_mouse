#!/usr/bin/env python
"""
Virtual Mouse - Control your mouse with hand gestures

This application uses computer vision to track hand movements and gestures,
allowing you to control your mouse pointer, perform clicks, scrolling, and
drag operations using just your hand in front of a webcam.
"""

import cv2
import numpy as np
import argparse
import config
from virtual_mouse import VirtualMouse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Virtual Mouse - Control your mouse with hand gestures")
    
    # Camera settings
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help=f"Camera index (default: {config.CAMERA_INDEX})")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH,
                        help=f"Camera width (default: {config.CAMERA_WIDTH})")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT,
                        help=f"Camera height (default: {config.CAMERA_HEIGHT})")
    
    # Tracking settings
    parser.add_argument("--detection-confidence", type=float, default=config.DETECTION_CONFIDENCE,
                        help=f"Hand detection confidence threshold (default: {config.DETECTION_CONFIDENCE})")
    parser.add_argument("--tracking-confidence", type=float, default=config.TRACKING_CONFIDENCE,
                        help=f"Hand tracking confidence threshold (default: {config.TRACKING_CONFIDENCE})")
    parser.add_argument("--max-hands", type=int, default=config.MAX_HANDS,
                        help=f"Maximum number of hands to detect (default: {config.MAX_HANDS})")
    
    # Mouse behavior
    parser.add_argument("--smoothing", type=int, default=config.SMOOTHING_FACTOR,
                        help=f"Mouse movement smoothing factor (default: {config.SMOOTHING_FACTOR})")
    parser.add_argument("--margin", type=int, default=config.MARGIN,
                        help=f"Margin from frame edges (default: {config.MARGIN})")
    
    # UI options
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug information display")
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Disable hand landmark display")
    
    # Screen coverage
    parser.add_argument("--coverage", type=float, default=config.SCREEN_COVERAGE_FACTOR,
                        help=f"Screen coverage factor (default: {config.SCREEN_COVERAGE_FACTOR})")
    
    # Display options
    parser.add_argument("--display-scale", type=float, default=0.7,
                        help="Scale factor for the camera preview window (0.1-1.0, default: 0.7)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Override config settings with command line arguments
    config.CAMERA_INDEX = args.camera
    config.CAMERA_WIDTH = args.width
    config.CAMERA_HEIGHT = args.height
    config.DETECTION_CONFIDENCE = args.detection_confidence
    config.TRACKING_CONFIDENCE = args.tracking_confidence
    config.MAX_HANDS = args.max_hands
    config.SMOOTHING_FACTOR = args.smoothing
    config.MARGIN = args.margin
    config.DEBUG = not args.no_debug
    config.SHOW_LANDMARKS = not args.no_landmarks
    config.SCREEN_COVERAGE_FACTOR = args.coverage
    
    return args


def print_instructions():
    """Print instructions to console."""
    print("=" * 60)
    print(" Virtual Mouse - Control your mouse with hand gestures")
    print("=" * 60)
    print("\nControls:")
    print("  - Move Cursor: Move your index finger")
    print("  - Left Click: Join index finger and thumb")
    print("  - Right Click: Form a peace sign (index and middle finger extended)")
    print("  - Scroll: Extend three fingers and move up/down or left/right")
    print("  - Drag: Pinch and hold for a moment")
    print("\nTips for better hand detection:")
    print("  - Use good lighting on your hand")
    print("  - Position your hand 30-60 cm from the camera")
    print("  - Use a plain background if possible")
    print("  - Make clear, distinct gestures")
    print("\nKeyboard shortcuts:")
    print("  - 'ESC': Quit the application")
    print("  - 'd': Toggle debug information display")
    print("  - 'h': Toggle hand landmark display")
    print("=" * 60)
    print("\nPress 'ESC' to quit the application at any time.\n")


def main():
    """Main function to run the virtual mouse application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print instructions
    print_instructions()
    
    try:
        # Ensure display scale is within valid range
        display_scale = max(0.1, min(args.display_scale, 1.0))
        
        # Create and run virtual mouse with the specified display scale
        virtual_mouse = VirtualMouse(display_scale=display_scale)
        virtual_mouse.run()
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()
        print("\nApplication closed.")


if __name__ == "__main__":
    main() 