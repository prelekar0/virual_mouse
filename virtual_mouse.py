import cv2
import numpy as np
import time
import argparse
from hand_tracking import HandTracker, GestureType
from mouse_controller import MouseController
import config

def display_ui_info(image, gesture_name, mode=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Display UI information on the image"""
    # Display settings
    color = (0, 255, 0)
    thickness = 2
    
    # Create a semi-transparent overlay for text background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 330), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    
    # Display current gesture
    cv2.putText(
        image, 
        f"Gesture: {gesture_name}", 
        (10, 30), 
        font, 
        0.7, 
        color, 
        thickness
    )
    
    # Display current mode if provided
    if mode:
        cv2.putText(
            image, 
            f"Mode: {mode}", 
            (10, 60), 
            font, 
            0.7, 
            color, 
            thickness
        )
    
    # Add usage instructions for gestures
    cv2.putText(
        image,
        "Open Hand = Neutral",
        (10, 90),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Peace Sign = Move Cursor",
        (10, 120),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Middle Finger Only = Left Click",
        (10, 150),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Index Finger Only = Right Click",
        (10, 180),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Index & Middle Fingers Close = Double Click",
        (10, 210),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Thumb + Index + Middle Up = Scroll Mode",
        (10, 240),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "- Move Up/Down for Vertical Scrolling",
        (30, 270),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "- Move Left/Right for Horizontal Scrolling",
        (30, 300),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Add keyboard shortcuts
    cv2.putText(
        image,
        "ESC = Exit | D = Toggle Debug | H = Toggle Landmarks",
        (10, 330),
        font,
        0.6,
        (255, 200, 100),
        1
    )
    
    return image

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Virtual Mouse using Hand Gestures")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--smoothing", type=float, default=0.5, help="Smoothing factor for cursor movement (0-1)")
    parser.add_argument("--cooldown", type=float, default=0.5, help="Cooldown between clicks in seconds")
    parser.add_argument("--show-landmarks", action="store_true", help="Show hand landmarks")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Initialize the camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera. Please check your camera connection.")
        return
    
    # Initialize hand tracker and mouse controller
    hand_tracker = HandTracker(
        detection_confidence=0.7,
        tracking_confidence=0.7
    )
    
    mouse_controller = MouseController(
        smoothing_factor=args.smoothing,
        click_cooldown=args.cooldown
    )
    
    # Print instructions
    print("\n===== Virtual Mouse Controller =====")
    print("Move your hand in the camera view to control the mouse cursor.")
    print("Gestures:")
    print("  - Open Hand: Neutral position (no action)")
    print("  - Peace Sign: Move cursor")
    print("  - Middle Finger Only: Left click")
    print("  - Index Finger Only: Right click")
    print("  - Peace Sign with fingers close together: Double click")
    print("  - Thumb + Index + Middle Up: Scroll mode")
    print("    * Move hand up/down for vertical scrolling")
    print("    * Move hand left/right for horizontal scrolling")
    print("Tips for better detection:")
    print("  - Place your hand against a plain background")
    print("  - Ensure good lighting on your hand")
    print("  - Keep gestures clear and distinct")
    print("  - Position your hand about 30-60 cm from the camera")
    print("Keyboard shortcuts:")
    print("  - Press ESC to exit")
    print("  - Press D to toggle debug mode")
    print("  - Press H to toggle hand landmarks")
    
    # Create window with trackbars if debug mode is enabled
    if args.debug:
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Smoothing', 'Controls', int(args.smoothing * 10), 20, 
                          lambda x: setattr(mouse_controller, 'smoothing_factor', max(1, x / 10)))
        cv2.createTrackbar('Gesture History', 'Controls', hand_tracker.gesture_history_length, 10, 
                          lambda x: setattr(hand_tracker, 'gesture_history_length', max(1, x)))
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # Last detected gesture for UI
    last_gesture = GestureType.UNKNOWN
    
    # Flags for display options
    show_debug = args.debug
    show_landmarks = args.show_landmarks
    
    # Main loop
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Mirror the image horizontally for a more intuitive interaction
        image = cv2.flip(image, 1)
        
        # Process the frame to find hands
        if show_landmarks:
            image, results = hand_tracker.find_hands(image, draw=True)
        else:
            _, results = hand_tracker.find_hands(image, draw=False)
        
        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # Default values for UI
        gesture_name = "None Detected"
        mode = "Idle"
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            # We're only tracking one hand, so use the first detection
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get gesture information
            gesture_info = hand_tracker.get_gesture_info(hand_landmarks)
            last_gesture = gesture_info["gesture"]
            
            # Set the gesture name for UI display
            gesture_name = last_gesture
            
            # Handle different gestures
            if last_gesture == GestureType.NEUTRAL:
                # Neutral position - do nothing
                mode = "Neutral"
                
            elif last_gesture == GestureType.CURSOR_MOVE:
                # Move cursor
                if gesture_info["cursor_position"]:
                    x, y = gesture_info["cursor_position"]
                    mouse_controller.smooth_move(x, y)
                    mode = "Cursor Movement"
                
            elif last_gesture == GestureType.LEFT_CLICK:
                # Left click
                if gesture_info["cursor_position"]:
                    x, y = gesture_info["cursor_position"]
                    mouse_controller.smooth_move(x, y)
                if mouse_controller.left_click():
                    mode = "Left Click"
                
            elif last_gesture == GestureType.RIGHT_CLICK:
                # Right click
                if gesture_info["cursor_position"]:
                    x, y = gesture_info["cursor_position"]
                    mouse_controller.smooth_move(x, y)
                if mouse_controller.right_click():
                    mode = "Right Click"
                    
            elif last_gesture == GestureType.DOUBLE_CLICK:
                # Double click - first move cursor then perform double click
                if gesture_info["cursor_position"]:
                    x, y = gesture_info["cursor_position"]
                    mouse_controller.smooth_move(x, y)
                if mouse_controller.double_click():
                    mode = "Double Click"
                    
            elif last_gesture == GestureType.SCROLL:
                # Scroll mode - handle both vertical and horizontal scrolling
                if gesture_info["cursor_position"]:
                    x, y = gesture_info["cursor_position"]
                    scrolled, direction = mouse_controller.handle_scroll(x, y)
                    if scrolled:
                        mode = f"Scrolling {direction.upper()}"
                    else:
                        mode = "Scroll Mode"
        
        # Display UI information
        image = display_ui_info(image, gesture_name, mode)
        
        # Add FPS counter if in debug mode
        if show_debug:
            cv2.putText(image, f"FPS: {int(fps)}", (image.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Virtual Mouse Control', image)
        
        # Show debug controls window if enabled
        if show_debug and not cv2.getWindowProperty('Controls', cv2.WND_PROP_VISIBLE):
            break
        
        # Check for key presses
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('d') or key == ord('D'):  # Toggle debug mode
            show_debug = not show_debug
            if show_debug:
                cv2.namedWindow('Controls')
                cv2.createTrackbar('Smoothing', 'Controls', int(mouse_controller.smoothing_factor * 10), 20, 
                                 lambda x: setattr(mouse_controller, 'smoothing_factor', max(1, x / 10)))
                cv2.createTrackbar('Gesture History', 'Controls', hand_tracker.gesture_history_length, 10, 
                                 lambda x: setattr(hand_tracker, 'gesture_history_length', max(1, x)))
            else:
                cv2.destroyWindow('Controls')
            print(f"Debug mode {'enabled' if show_debug else 'disabled'}")
        elif key == ord('h') or key == ord('H'):  # Toggle landmarks display
            show_landmarks = not show_landmarks
            print(f"Landmarks display {'enabled' if show_landmarks else 'disabled'}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nVirtual Mouse Controller closed. Thank you for using!")

if __name__ == "__main__":
    main() 