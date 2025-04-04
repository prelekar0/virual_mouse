import cv2
import numpy as np
import time
import argparse
from hand_tracking import HandTracker, GestureType
from mouse_controller import MouseController
import config
import pyautogui
import math

def display_ui_info(image, gesture_name, mode=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Display UI information on the image"""
    # Display settings
    color = (0, 255, 0)
    thickness = 2
    
    # Create a semi-transparent overlay for text background
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (350, 400), (0, 0, 0), -1)
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
        "Index & Middle Fingers Up = Move Cursor",
        (10, 120),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Index Down, Middle Up = Left Click",
        (10, 150),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Index Up, Middle Down = Right Click",
        (10, 180),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "From Neutral, Lower Index = Left Click",
        (10, 210),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "From Neutral, Lower Middle = Right Click",
        (10, 240),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Index & Middle Fingers Close = Double Click",
        (10, 270),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Thumb + Index Pinch = Drag & Drop",
        (10, 300),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "Thumb + Index + Middle Up = Scroll Mode",
        (10, 330),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    cv2.putText(
        image,
        "- Move Up/Down/Left/Right for Scrolling",
        (30, 360),
        font,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Add keyboard shortcuts
    cv2.putText(
        image,
        "ESC = Exit | D = Toggle Debug | H = Toggle Landmarks",
        (10, 390),
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
    parser.add_argument("--display-scale", type=float, default=0.7, help="Scale factor for display window (0.1-1.0)")
    
    return parser.parse_args()

class VirtualMouse:
    """
    Main class for the Virtual Mouse application.
    
    This class integrates hand tracking with mouse control to create
    a virtual mouse controlled by hand gestures.
    """
    
    def __init__(self, cap=None, show_debug=False, show_landmarks=True, display_scale=0.75):
        """Initialize the virtual mouse controller."""
        # Initialize camera if not provided
        if cap is None:
            self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            
            # Set frame rate if available
            if hasattr(config, 'CAMERA_FPS'):
                self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        else:
            self.cap = cap
        
        # Initialize display settings
        self.show_debug = show_debug
        self.show_landmarks = show_landmarks
        self.display_scale = display_scale
        
        # Initialize hand tracker
        self.hand_tracker = HandTracker(
            detection_confidence=config.DETECTION_CONFIDENCE,
            tracking_confidence=config.TRACKING_CONFIDENCE,
            max_hands=config.MAX_HANDS
        )
        
        # Initialize mouse controller
        self.mouse_controller = MouseController(
            smoothing_factor=config.SMOOTHING_FACTOR,
            click_cooldown=config.CLICK_COOLDOWN
        )
        
        # Initialize other attributes
        self.mode = None
        self.last_frame_time = 0
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Get screen dimensions for reference
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen dimensions: {self.screen_width}x{self.screen_height}")
        
        # Debug display window setup
        self.debug_window_setup = False
        
    def run(self):
        """Run the virtual mouse application."""
        if not self.cap.isOpened():
            print("Error: Could not open camera. Please check your camera connection.")
            return
        
        print("\n===== Virtual Mouse Controller =====")
        print("Move your hand in the camera view to control the mouse cursor.")
        print("Gestures:")
        print("  - Open Hand: Neutral position (no action)")
        print("  - Index & Middle Fingers Up: Move cursor")
        print("  - Index Down, Middle Up: Left click")
        print("  - Index Up, Middle Down: Right click")
        print("  - From Neutral, Lower Index Finger: Left click at current position")
        print("  - From Neutral, Lower Middle Finger: Right click at current position")
        print("  - Index & Middle Fingers Close: Double click")
        print("  - Thumb + Index Pinch: Drag & Drop")
        print("  - Thumb + Index + Middle Up: Scroll mode")
        print("    * Move hand up/down for vertical scrolling")
        print("    * Move hand left/right for horizontal scrolling")
        print("Keyboard shortcuts:")
        print("  - Press ESC to exit")
        print("  - Press D to toggle debug mode")
        print("  - Press H to toggle hand landmarks")
        
        # Create control window if in debug mode
        if self.show_debug:
            self._setup_debug_window()
        
        # Variables for performance optimization
        last_process_time = time.time()
        process_interval = 1.0 / config.MAX_FPS if hasattr(config, 'MAX_FPS') else 0
        skip_frames = config.PROCESS_EVERY_N_FRAMES if hasattr(config, 'PROCESS_EVERY_N_FRAMES') else 1
        frame_count = 0
        
        # Variable to store the last valid cursor position for click-without-move feature
        last_valid_cursor_position = None
        previous_gesture = None
        
        # Main loop
        while True:
            # Read frame from camera
            success, image = self.cap.read()
            if not success:
                print("Failed to read from camera")
                break
            
            # Performance optimization: skip frames if needed
            frame_count += 1
            current_time = time.time()
            should_process = (frame_count % skip_frames == 0) and (current_time - last_process_time >= process_interval)
            
            if not should_process:
                # Just display the image without processing
                cv2.imshow('Virtual Mouse Control', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                    break
                continue
            
            last_process_time = current_time
                
            # Mirror the image horizontally
            image = cv2.flip(image, 1)
            
            # Apply image enhancements for better hand detection if enabled
            if hasattr(config, 'BOOST_CONTRAST') and config.BOOST_CONTRAST:
                image = self._enhance_image(image)
            
            # Process the frame to find hands
            if self.show_landmarks:
                image, results = self.hand_tracker.find_hands(image, draw=True)
            else:
                _, results = self.hand_tracker.find_hands(image, draw=False)
            
            # Update FPS
            self._update_fps()
            
            # Reset mode and gesture
            gesture_name = "None Detected"
            self.mode = "Idle"
            
            # Process hand gesture if hand is detected
            if results.multi_hand_landmarks:
                # We only use the first hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Get gesture and cursor position
                gesture, gesture_info = self.hand_tracker.recognize_hand_gesture(hand_landmarks)
                
                # Set gesture name for display
                gesture_name = gesture
                
                # Handle different gestures
                if gesture == GestureType.UNKNOWN:
                    self.mode = "Unknown Gesture"
                
                elif gesture == GestureType.NEUTRAL:
                    self.mode = "Neutral"
                    # In neutral mode, we don't move the cursor but keep the last valid position
                    # Store the last position from the cursor_move for later clicks
                
                elif gesture == GestureType.CURSOR_MOVE:
                    # Move cursor with adaptive control based on distance
                    if gesture_info["cursor_position"]:
                        x, y = gesture_info["cursor_position"]
                        
                        # Debug info in debug mode
                        if self.show_debug:
                            print(f"CURSOR_MOVE detected - Position: ({x:.3f}, {y:.3f})")
                        
                        # Calculate more precise cursor position using both index and middle fingertips
                        # This gives better stability than just using the index finger
                        if len(hand_landmarks.landmark) >= 12:  # Ensure we have enough landmarks
                            index_tip = hand_landmarks.landmark[8]
                            middle_tip = hand_landmarks.landmark[12]
                            index_pip = hand_landmarks.landmark[6]  # Index PIP joint
                            
                            # For better accuracy, primarily use the index finger with just a small
                            # amount of middle finger influence for stability
                            x = index_tip.x * 0.9 + middle_tip.x * 0.1
                            y = index_tip.y * 0.9 + middle_tip.y * 0.1
                            
                            # Get distance factor for speed adjustments
                            distance_factor = gesture_info.get("distance_factor", 1.0)
                            
                            # Apply basic smoothing with position history
                            prev_x, prev_y = self.mouse_controller.get_normalized_position()
                            if prev_x is not None and prev_y is not None:
                                # Use a 2-stage smoothing approach:
                                # 1. Apply initial smoothing to reduce micro-movements
                                x = prev_x * 0.3 + x * 0.7
                                y = prev_y * 0.3 + y * 0.7
                                
                                # 2. Calculate movement speed for acceleration
                                delta_x = x - prev_x
                                delta_y = y - prev_y
                                move_distance = math.sqrt(delta_x**2 + delta_y**2)
                                
                                # Apply acceleration for faster movements
                                if move_distance > 0.03:  # Larger, intentional movement
                                    # Boost the movement by amplifying the delta
                                    acceleration = min(2.5, 1.0 + move_distance * 15)
                                    x = prev_x + delta_x * acceleration
                                    y = prev_y + delta_y * acceleration
                                    
                                    # Ensure coordinates stay within bounds
                                    x = max(0.0, min(1.0, x))
                                    y = max(0.0, min(1.0, y))
                            
                            # Send the final position to the mouse controller
                            self.mouse_controller.smooth_move(x, y)
                            last_valid_cursor_position = (x, y)
                            self.mode = "Cursor Movement"
                        else:
                            # Fallback to original position if landmarks unavailable
                            self.mouse_controller.smooth_move(x, y)
                            last_valid_cursor_position = (x, y)
                            self.mode = "Cursor Movement"
                
                elif gesture == GestureType.LEFT_CLICK:
                    # Perform click at current or last valid position
                    self.mode = "LEFT CLICK"
                    
                    if last_valid_cursor_position:
                        # Use the last valid cursor position without moving
                        x, y = last_valid_cursor_position
                    else:
                        # Use the current position from gesture_info as fallback
                        x, y = gesture_info["cursor_position"] if gesture_info["cursor_position"] else (0.5, 0.5)
                    
                    # Perform click without moving the cursor
                    if self.mouse_controller.click():
                        # Convert normalized coordinates to screen coordinates for display
                        screen_x = int(x * image.shape[1])
                        screen_y = int(y * image.shape[0])
                        
                        # Draw click feedback
                        image = self._draw_click_feedback(image, screen_x, screen_y, (0, 255, 0))
                        
                        # Debug info
                        if self.show_debug:
                            cursor_x, cursor_y = self.mouse_controller.prev_x, self.mouse_controller.prev_y
                            print(f"Click performed at: ({int(cursor_x)}, {int(cursor_y)})")
                
                elif gesture == GestureType.RIGHT_CLICK:
                    # Perform right click at current or last valid position
                    self.mode = "RIGHT CLICK"
                    
                    if last_valid_cursor_position:
                        # Use the last valid cursor position without moving
                        x, y = last_valid_cursor_position
                    else:
                        # Use the current position from gesture_info as fallback
                        x, y = gesture_info["cursor_position"] if gesture_info["cursor_position"] else (0.5, 0.5)
                    
                    # Perform right click without moving the cursor
                    if self.mouse_controller.right_click():
                        # Convert normalized coordinates to screen coordinates for display
                        screen_x = int(x * image.shape[1])
                        screen_y = int(y * image.shape[0])
                        
                        # Draw click feedback
                        image = self._draw_click_feedback(image, screen_x, screen_y, (255, 0, 0))
                        
                        # Debug info
                        if self.show_debug:
                            cursor_x, cursor_y = self.mouse_controller.prev_x, self.mouse_controller.prev_y
                            print(f"Right click performed at: ({int(cursor_x)}, {int(cursor_y)})")
                
                elif gesture == GestureType.DOUBLE_CLICK:
                    # Perform double click at current position
                    self.mode = "DOUBLE CLICK"
                    
                    if last_valid_cursor_position:
                        # Use the last valid cursor position without moving
                        x, y = last_valid_cursor_position
                    else:
                        # Use the current position from gesture_info as fallback
                        x, y = gesture_info["cursor_position"] if gesture_info["cursor_position"] else (0.5, 0.5)
                    
                    # Perform double click without moving the cursor
                    if self.mouse_controller.double_click():
                        # Convert normalized coordinates to screen coordinates for display
                        screen_x = int(x * image.shape[1])
                        screen_y = int(y * image.shape[0])
                        
                        # Draw click feedback
                        image = self._draw_click_feedback(image, screen_x, screen_y, (0, 255, 255), text="DOUBLE CLICK")
                        
                        # Debug info
                        if self.show_debug:
                            print("Performing double-click")
                
                elif gesture == GestureType.DRAG:
                    # Handle drag gesture - start, continue, or end drag
                    self.mode = "DRAG"
                    
                    if gesture_info["cursor_position"]:
                        x, y = gesture_info["cursor_position"]
                        
                        # Debug info in debug mode
                        if self.show_debug:
                            print(f"DRAG mode detected - Position: ({x:.3f}, {y:.3f})")
                        
                        # Get more stable position by using weighted average of multiple points
                        if len(hand_landmarks.landmark) >= 9:  # Ensure we have enough landmarks
                            # Use index finger base for more stability during dragging
                            index_mcp = hand_landmarks.landmark[5]  # Index MCP joint
                            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                            index_tip = hand_landmarks.landmark[8]  # Index tip
                            
                            # Calculate weighted position between index base and the pinch point
                            pinch_point_x = (thumb_tip.x + index_tip.x) / 2
                            pinch_point_y = (thumb_tip.y + index_tip.y) / 2
                            
                            # Weight more towards the pinch point
                            x = pinch_point_x * 0.7 + index_mcp.x * 0.3
                            y = pinch_point_y * 0.7 + index_mcp.y * 0.3
                            
                            # Apply very slight smoothing for stability
                            prev_x, prev_y = self.mouse_controller.get_normalized_position()
                            if prev_x is not None and prev_y is not None:
                                x = x * 0.8 + prev_x * 0.2
                                y = y * 0.8 + prev_y * 0.2
                        
                        # Convert to screen coordinates for mouse controller
                        screen_x = int(x * self.screen_width)
                        screen_y = int(y * self.screen_height)
                        
                        # Ensure coordinates are within screen bounds
                        screen_x = max(0, min(screen_x, self.screen_width - 1))
                        screen_y = max(0, min(screen_y, self.screen_height - 1))
                        
                        # Check drag state
                        if not self.mouse_controller.is_dragging:
                            # Start drag operation
                            self.mouse_controller.start_drag(screen_x, screen_y)
                            
                            # Debug info
                            if self.show_debug:
                                print(f"Started drag at: ({screen_x}, {screen_y})")
                                
                            # Draw drag start indicator
                            image = self._draw_drag_feedback(image, int(x * image.shape[1]), int(y * image.shape[0]), 
                                                          (255, 165, 0), "DRAG START")
                        else:
                            # Continue drag operation
                            self.mouse_controller.continue_drag(screen_x, screen_y)
                            
                            # Draw drag indicator
                            image = self._draw_drag_feedback(image, int(x * image.shape[1]), int(y * image.shape[0]), 
                                                          (255, 165, 0), "DRAGGING")
                
                elif gesture == GestureType.SCROLL:
                    # Scroll mode with adaptive speed based on distance
                    if gesture_info["cursor_position"]:
                        x, y = gesture_info["cursor_position"]
                        
                        # Debug info in debug mode
                        if self.show_debug:
                            print(f"SCROLL mode detected - Position: ({x:.3f}, {y:.3f})")
                        
                        # Get a more stable position for scrolling - use the palm center
                        if len(hand_landmarks.landmark) >= 21:
                            # Calculate palm center from wrist and middle of palm
                            wrist = hand_landmarks.landmark[0]  # Wrist landmark
                            palm_center = hand_landmarks.landmark[9]  # Middle of palm landmark
                            
                            # Weight palm center more (70%) for stability
                            scroll_x = palm_center.x * 0.7 + wrist.x * 0.3
                            scroll_y = palm_center.y * 0.7 + wrist.y * 0.3
                            
                            # Reset scroll coordinates if they haven't been set
                            if self.mouse_controller.prev_scroll_x is None or self.mouse_controller.prev_scroll_y is None:
                                self.mouse_controller.prev_scroll_x = scroll_x
                                self.mouse_controller.prev_scroll_y = scroll_y
                                time.sleep(0.05)  # Brief pause for stability
                            
                            # Set scroll speed based on hand distance
                            distance_factor = gesture_info.get("distance_factor", 1.0)
                            self.mouse_controller.scroll_speed = max(2, int(config.SCROLL_SPEED * distance_factor))
                            
                            # Handle scroll
                            scrolled, direction = self.mouse_controller.handle_scroll(scroll_x, scroll_y)
                            
                            if scrolled:
                                self.mode = f"Scrolling {direction.upper()}"
                                
                                # Draw direction indicator on screen
                                h, w, _ = image.shape
                                center_x, center_y = w // 2, h // 2
                                arrow_size = 50
                                arrow_color = (0, 255, 255)  # Yellow
                                
                                if direction == "up":
                                    cv2.arrowedLine(image, (center_x, center_y + arrow_size), 
                                                  (center_x, center_y - arrow_size), arrow_color, 4)
                                elif direction == "down":
                                    cv2.arrowedLine(image, (center_x, center_y - arrow_size), 
                                                  (center_x, center_y + arrow_size), arrow_color, 4)
                                elif direction == "left":
                                    cv2.arrowedLine(image, (center_x + arrow_size, center_y), 
                                                  (center_x - arrow_size, center_y), arrow_color, 4)
                                elif direction == "right":
                                    cv2.arrowedLine(image, (center_x - arrow_size, center_y), 
                                                  (center_x + arrow_size, center_y), arrow_color, 4)
                                
                                # Debug info
                                if self.show_debug:
                                    print(f"Scrolling {direction}")
                            else:
                                self.mode = "Scroll Mode"
                        else:
                            self.mode = "Scroll Mode"
                
                # End drag if changing from drag to another gesture
                if previous_gesture == GestureType.DRAG and gesture != GestureType.DRAG:
                    if self.mouse_controller.is_dragging:
                        # Get current position
                        current_pos = self.mouse_controller.get_normalized_position()
                        if current_pos[0] is not None:
                            # Convert to screen coordinates
                            screen_x = int(current_pos[0] * self.screen_width)
                            screen_y = int(current_pos[1] * self.screen_height)
                            
                            # End drag operation
                            self.mouse_controller.stop_drag(screen_x, screen_y)
                            
                            # Debug info
                            if self.show_debug:
                                print(f"Ended drag at: ({screen_x}, {screen_y})")
                
                # Reset scroll positions when not in scroll mode
                if previous_gesture == GestureType.SCROLL and gesture != GestureType.SCROLL:
                    self.mouse_controller.prev_scroll_x = None
                    self.mouse_controller.prev_scroll_y = None
                
                # Store the current gesture as previous for next frame
                previous_gesture = gesture
            
            # Display UI information
            image = display_ui_info(image, gesture_name, self.mode)
            
            # Add FPS counter if in debug mode
            if self.show_debug:
                cv2.putText(image, f"FPS: {int(self.fps)}", (image.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Resize the frame for display
            display_width = int(image.shape[1] * self.display_scale)
            display_height = int(image.shape[0] * self.display_scale)
            display_image = cv2.resize(image, (display_width, display_height))
            
            # Display the frame
            cv2.imshow('Virtual Mouse Control', display_image)
            
            # Check for key presses
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord('d') or key == ord('D'):  # Toggle debug mode
                self._toggle_debug_mode()
            elif key == ord('h') or key == ord('H'):  # Toggle landmarks display
                self.show_landmarks = not self.show_landmarks
                print(f"Landmarks display {'enabled' if self.show_landmarks else 'disabled'}")
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nVirtual Mouse Controller closed.")
    
    def _enhance_image(self, image):
        """Enhance image contrast and brightness for better hand detection."""
        # Apply contrast and brightness adjustments
        enhanced = cv2.convertScaleAbs(image, alpha=config.CONTRAST_ALPHA, beta=config.CONTRAST_BETA)
        return enhanced
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        if (current_time - self.fps_start_time) > 1:
            self.fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _setup_debug_window(self):
        """Set up debug controls window."""
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Smoothing', 'Controls', int(self.mouse_controller.smoothing_factor), 20, 
                         lambda x: setattr(self.mouse_controller, 'smoothing_factor', max(1, x / 10)))
        cv2.createTrackbar('Margin', 'Controls', config.MARGIN, 150, 
                         lambda x: setattr(config, 'MARGIN', max(10, x)))
        cv2.createTrackbar('Click Cooldown', 'Controls', int(config.CLICK_COOLDOWN * 10), 10, 
                         lambda x: setattr(config, 'CLICK_COOLDOWN', max(0.1, x / 10)))
    
    def _toggle_debug_mode(self):
        """Toggle debug mode on/off."""
        self.show_debug = not self.show_debug
        if self.show_debug:
            self._setup_debug_window()
        else:
            cv2.destroyWindow('Controls')
        print(f"Debug mode {'enabled' if self.show_debug else 'disabled'}")

    def _draw_click_feedback(self, image, x, y, color=(0, 0, 255), radius=20, duration_frames=5, text="CLICK"):
        """Draw visual feedback for clicks on the image."""
        # Create a copy of the image to avoid modifying the original
        feedback_image = image.copy()
        
        # Draw animated click indicator with concentric circles
        cv2.circle(feedback_image, (x, y), radius, color, 2)
        cv2.circle(feedback_image, (x, y), radius//2, color, 1)
        cv2.circle(feedback_image, (x, y), 6, color, cv2.FILLED)
        
        # Add a text indicator
        cv2.putText(feedback_image, text, (x + radius + 5, y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Apply the same scaling factor as the main display
        display_scale = self.display_scale  # Must match the scale in run() method
        display_width = int(feedback_image.shape[1] * display_scale)
        display_height = int(feedback_image.shape[0] * display_scale)
        display_feedback = cv2.resize(feedback_image, (display_width, display_height))
        
        # Display the feedback image
        cv2.imshow('Virtual Mouse Control', display_feedback)
        cv2.waitKey(1)  # Brief display
        
        return feedback_image

    def _draw_drag_feedback(self, image, x, y, color=(0, 0, 255), text="DRAG"):
        """Draw visual feedback for drag gestures on the image."""
        # Create a copy of the image to avoid modifying the original
        feedback_image = image.copy()
        
        # Draw drag indicator
        cv2.rectangle(feedback_image, (x - 20, y - 20), (x + 20, y + 20), color, 2)
        
        # Add text indicator
        cv2.putText(feedback_image, text, (x - 20, y - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Apply the same scaling factor as the main display
        display_scale = self.display_scale  # Must match the scale in run() method
        display_width = int(feedback_image.shape[1] * display_scale)
        display_height = int(feedback_image.shape[0] * display_scale)
        display_feedback = cv2.resize(feedback_image, (display_width, display_height))
        
        # Display the feedback image
        cv2.imshow('Virtual Mouse Control', display_feedback)
        cv2.waitKey(1)  # Brief display
        
        return feedback_image

def main():
    """Main function to run the Virtual Mouse application."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Update config with command-line arguments if provided
    if args.camera is not None:
        config.CAMERA_INDEX = args.camera
    if args.width is not None:
        config.CAMERA_WIDTH = args.width
    if args.height is not None:
        config.CAMERA_HEIGHT = args.height
    if args.smoothing is not None:
        config.SMOOTHING_FACTOR = args.smoothing
    if args.cooldown is not None:
        config.CLICK_COOLDOWN = args.cooldown
    if args.show_landmarks:
        config.SHOW_LANDMARKS = True
    if args.debug:
        config.DEBUG = True
    
    # Create and run the virtual mouse with the specified display scale
    display_scale = max(0.1, min(args.display_scale, 1.0))  # Ensure scale is between 0.1 and 1.0
    vm = VirtualMouse(display_scale=display_scale)
    vm.run()
    print("Application closed.")

if __name__ == "__main__":
    main() 