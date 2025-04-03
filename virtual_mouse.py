#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse
from hand_tracker import HandTracker, Gesture
from mouse_controller import MouseController

def display_ui_info(image, gesture_name, mode=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Display UI information on the image"""
    # Display settings
    color = (0, 255, 0)
    thickness = 2
    
    # Create a semi-transparent overlay for text background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 130), (0, 0, 0), -1)
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
    
    # Add usage instructions for ESC key
    cv2.putText(
        image,
        "Press ESC to exit",
        (10, 90),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Add suggestion for better hand detection
    cv2.putText(
        image,
        "Use plain background for better detection",
        (10, 120),
        font,
        0.6,
        (255, 100, 100),
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
    parser.add_argument("--skin-calibration", action="store_true", help="Enable skin color calibration mode")
    
    return parser.parse_args()

def adjust_skin_color(hand_tracker, hsv_frame, x, y, w, h):
    """Adjust skin color thresholds based on a region of the hand"""
    # Take a sample from the center of the hand region
    sample_region = hsv_frame[y+h//4:y+3*h//4, x+w//4:x+3*w//4]
    
    if sample_region.size > 0:
        # Calculate mean and std of HSV in the hand region
        h_mean, s_mean, v_mean = np.mean(sample_region, axis=(0, 1))
        h_std, s_std, v_std = np.std(sample_region, axis=(0, 1))
        
        # Set new thresholds with margins
        h_lower = max(0, h_mean - 2*h_std)
        h_upper = min(180, h_mean + 2*h_std)
        s_lower = max(0, s_mean - 2*s_std)
        s_upper = min(255, s_mean + 2*s_std)
        v_lower = max(0, v_mean - 2*v_std)
        v_upper = min(255, v_mean + 2*v_std)
        
        # Update tracker thresholds
        hand_tracker.lower_skin = np.array([h_lower, s_lower, v_lower], dtype=np.uint8)
        hand_tracker.upper_skin = np.array([h_upper, s_upper, v_upper], dtype=np.uint8)
        
        return True
    
    return False

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
        min_detection_confidence=0.5,  # Lowered for better sensitivity
        min_tracking_confidence=0.5    # Lowered for better sensitivity
    )
    
    mouse_controller = MouseController(
        smoothing_factor=args.smoothing,
        click_cooldown=args.cooldown
    )
    
    # Print instructions
    print("\n===== Virtual Mouse Controller =====")
    print("Move your hand in the camera view to control the mouse cursor.")
    print("Gestures:")
    print("  - Move hand: Move cursor")
    print("  - Fist: Left click")
    print("  - Two fingers extended: Right click")
    print("  - Open palm: Scroll mode")
    print("Tips for better detection:")
    print("  - Place your hand against a plain background")
    print("  - Ensure good lighting on your hand")
    print("  - Move slowly at first to help detection")
    print("  - Keep your hand in the frame")
    print("Press ESC to exit\n")
    
    # Create window with trackbars if debug mode is enabled
    if args.debug:
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Min Area', 'Controls', hand_tracker.hand_area_min, 20000, lambda x: setattr(hand_tracker, 'hand_area_min', max(500, x)))
        cv2.createTrackbar('Max Area', 'Controls', hand_tracker.hand_area_max, 200000, lambda x: setattr(hand_tracker, 'hand_area_max', max(20000, x)))
        
        # Add skin color threshold controls
        cv2.createTrackbar('H Min', 'Controls', hand_tracker.lower_skin[0], 180, lambda x: setattr(hand_tracker, 'lower_skin', np.array([x, hand_tracker.lower_skin[1], hand_tracker.lower_skin[2]], dtype=np.uint8)))
        cv2.createTrackbar('H Max', 'Controls', hand_tracker.upper_skin[0], 180, lambda x: setattr(hand_tracker, 'upper_skin', np.array([x, hand_tracker.upper_skin[1], hand_tracker.upper_skin[2]], dtype=np.uint8)))
        cv2.createTrackbar('S Min', 'Controls', hand_tracker.lower_skin[1], 255, lambda x: setattr(hand_tracker, 'lower_skin', np.array([hand_tracker.lower_skin[0], x, hand_tracker.lower_skin[2]], dtype=np.uint8)))
        cv2.createTrackbar('S Max', 'Controls', hand_tracker.upper_skin[1], 255, lambda x: setattr(hand_tracker, 'upper_skin', np.array([hand_tracker.upper_skin[0], x, hand_tracker.upper_skin[2]], dtype=np.uint8)))
        cv2.createTrackbar('V Min', 'Controls', hand_tracker.lower_skin[2], 255, lambda x: setattr(hand_tracker, 'lower_skin', np.array([hand_tracker.lower_skin[0], hand_tracker.lower_skin[1], x], dtype=np.uint8)))
        cv2.createTrackbar('V Max', 'Controls', hand_tracker.upper_skin[2], 255, lambda x: setattr(hand_tracker, 'upper_skin', np.array([hand_tracker.upper_skin[0], hand_tracker.upper_skin[1], x], dtype=np.uint8)))
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # Skin color calibration mode
    calibration_mode = args.skin_calibration
    calibration_rect = None  # Rectangle for skin color sampling
    
    # Main loop
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        
        # Mirror the image horizontally for a more intuitive interaction
        image = cv2.flip(image, 1)
        
        # Skin color calibration mode
        if calibration_mode:
            h, w = image.shape[:2]
            rect_size = min(w, h) // 4
            x = w // 2 - rect_size // 2
            y = h // 2 - rect_size // 2
            calibration_rect = (x, y, rect_size, rect_size)
            
            # Convert to HSV for skin color calibration
            hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Draw rectangle for hand placement
            cv2.rectangle(image, (x, y), (x + rect_size, y + rect_size), (0, 255, 0), 2)
            cv2.putText(image, "Place hand in box, press SPACE", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Skin Color Calibration', image)
            key = cv2.waitKey(5) & 0xFF
            
            if key == 32:  # SPACE key
                if adjust_skin_color(hand_tracker, hsv_frame, x, y, rect_size, rect_size):
                    print(f"Skin color calibrated - H: [{hand_tracker.lower_skin[0]}-{hand_tracker.upper_skin[0]}], " +
                          f"S: [{hand_tracker.lower_skin[1]}-{hand_tracker.upper_skin[1]}], " +
                          f"V: [{hand_tracker.lower_skin[2]}-{hand_tracker.upper_skin[2]}]")
                    
                    # Update trackbars if debug mode
                    if args.debug:
                        cv2.setTrackbarPos('H Min', 'Controls', hand_tracker.lower_skin[0])
                        cv2.setTrackbarPos('H Max', 'Controls', hand_tracker.upper_skin[0])
                        cv2.setTrackbarPos('S Min', 'Controls', hand_tracker.lower_skin[1])
                        cv2.setTrackbarPos('S Max', 'Controls', hand_tracker.upper_skin[1])
                        cv2.setTrackbarPos('V Min', 'Controls', hand_tracker.lower_skin[2])
                        cv2.setTrackbarPos('V Max', 'Controls', hand_tracker.upper_skin[2])
                    
                    calibration_mode = False
                    cv2.destroyWindow('Skin Color Calibration')
            
            elif key == 27:  # ESC key
                calibration_mode = False
                cv2.destroyWindow('Skin Color Calibration')
            
            continue
        
        # Process the frame with hand tracker
        results = hand_tracker.process_frame(image)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS
        cv2.putText(image, f"FPS: {fps}", (image.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Default UI information
        gesture_name = "No hand detected"
        mode = None
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            # We're only tracking one hand, so use the first detection
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw hand landmarks if enabled
            if args.show_landmarks:
                image = hand_tracker.draw_landmarks(image, hand_landmarks)
            
            # Get the index finger tip landmark for cursor position
            index_finger_tip = hand_landmarks.landmark[8]
            
            # Smooth the coordinates
            x, y = hand_tracker.smooth_position(index_finger_tip)
            
            # Detect gesture
            finger_states = hand_tracker.get_finger_states(hand_landmarks)
            gesture = hand_tracker.detect_gesture(finger_states)
            stable_gesture = hand_tracker.get_stable_gesture(gesture)
            
            # Set the gesture name for UI display
            gesture_name = stable_gesture.name
            
            # Handle different gestures
            if stable_gesture == Gesture.FIST:
                # Left click
                if mouse_controller.left_click():
                    gesture_name = "LEFT CLICK"
            
            elif stable_gesture == Gesture.TWO_FINGERS:
                # Right click
                if mouse_controller.right_click():
                    gesture_name = "RIGHT CLICK"
            
            elif stable_gesture == Gesture.OPEN_PALM:
                # Scroll mode
                screen_x, screen_y = mouse_controller.move_cursor(x, y, duration=0)
                action_performed, direction = mouse_controller.scroll(screen_x, screen_y)
                
                if action_performed:
                    gesture_name = f"SCROLLING {direction.upper()}"
                else:
                    gesture_name = "SCROLL MODE"
                
                mode = "Scrolling"
            
            else:  # POINTING or other gestures
                # Regular cursor movement
                if not mouse_controller.scroll_mode:
                    mouse_controller.move_cursor(x, y)
                    mode = "Cursor Movement"
        else:
            # No hand detected, exit scroll mode
            mouse_controller.exit_scroll_mode()
        
        # Display UI information
        image = display_ui_info(image, gesture_name, mode)
        
        # Display the frame
        cv2.imshow('Virtual Mouse Control', image)
        
        # Show debug controls window if enabled
        if args.debug and not cv2.getWindowProperty('Controls', cv2.WND_PROP_VISIBLE):
            break
        
        # Exit on ESC key, C key for calibration
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == 99:  # 'c' key
            print("Entering skin color calibration mode. Place your hand in the box and press SPACE.")
            calibration_mode = True
        
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    hand_tracker.release()
    
    print("\nVirtual Mouse Controller closed. Thank you for using!")

if __name__ == '__main__':
    main() 