import cv2
import mediapipe as mp
import numpy as np
import math
import config

class GestureType:
    NEUTRAL = "NEUTRAL"         # Open hand (all fingers up)
    CURSOR_MOVE = "CURSOR_MOVE" # Peace sign (index and middle fingers up)
    LEFT_CLICK = "LEFT_CLICK"   # Middle finger up, index finger down
    RIGHT_CLICK = "RIGHT_CLICK" # Index finger up, middle finger down
    DOUBLE_CLICK = "DOUBLE_CLICK" # Index and middle fingers close together
    SCROLL = "SCROLL"           # Three fingers up (thumb, index, middle)
    UNKNOWN = "UNKNOWN"         # Unrecognized gesture

class HandTracker:
    def __init__(self, static_mode=None, max_hands=None, detection_confidence=None, tracking_confidence=None):
        """
        Initialize the HandTracker with MediaPipe Hands.
        
        Args:
            static_mode (bool, optional): If set to False, the solution treats the input images as a video stream.
            max_hands (int, optional): Maximum number of hands to detect.
            detection_confidence (float, optional): Minimum confidence value for hand detection.
            tracking_confidence (float, optional): Minimum confidence value for hand landmarks tracking.
        """
        # Use config values if not specified
        self.static_mode = static_mode if static_mode is not None else config.STATIC_MODE
        self.max_hands = max_hands if max_hands is not None else config.MAX_HANDS
        self.detection_confidence = detection_confidence if detection_confidence is not None else config.DETECTION_CONFIDENCE
        self.tracking_confidence = tracking_confidence if tracking_confidence is not None else config.TRACKING_CONFIDENCE
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            model_complexity=1  # Use more accurate model (0, 1, or 2)
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Landmark smoothing
        self.landmark_history = []
        self.history_length = config.LANDMARK_HISTORY_LENGTH
        
        # Gesture history for stability
        self.gesture_history = []
        self.gesture_history_length = 5  # Frames to consider for stable gesture
        
        # Debug
        print(f"Hand Tracker initialized with detection confidence: {self.detection_confidence}, " 
              f"tracking confidence: {self.tracking_confidence}")
        
    def find_hands(self, frame, draw=True):
        """
        Process a frame to find hands and optionally draw landmarks.
        
        Args:
            frame (numpy.ndarray): Input image frame.
            draw (bool): Whether to draw hand landmarks on the frame.
            
        Returns:
            numpy.ndarray: Frame with or without drawn landmarks.
            dict: Results from MediaPipe hand processing.
        """
        # Store frame shape for coordinate conversions
        self.frame_shape = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to find hands
        results = self.hands.process(frame_rgb)
        
        # Draw landmarks if requested and hands are detected
        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get the current gesture
                finger_states = self.get_finger_states(hand_landmarks)
                gesture = self.recognize_gesture(finger_states)
                
                # Add visual indication of gesture
                h, w, c = frame.shape
                wrist = hand_landmarks.landmark[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                
                # Different colors for different gestures
                color = (255, 255, 255)  # Default white
                if gesture == GestureType.NEUTRAL:
                    color = (0, 255, 0)  # Green for neutral
                elif gesture == GestureType.CURSOR_MOVE:
                    color = (255, 0, 0)  # Blue for cursor move
                elif gesture == GestureType.LEFT_CLICK:
                    color = (0, 0, 255)  # Red for left click
                elif gesture == GestureType.RIGHT_CLICK:
                    color = (255, 255, 0)  # Cyan for right click
                elif gesture == GestureType.DOUBLE_CLICK:
                    color = (255, 0, 255)  # Magenta for double click
                elif gesture == GestureType.SCROLL:
                    color = (0, 165, 255)  # Orange for scroll
                
                # Draw gesture label
                cv2.putText(frame, gesture, (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, color, 2, cv2.LINE_AA)
                
                # Highlight index and middle fingers for better visibility
                index_tip = hand_landmarks.landmark[8]
                middle_tip = hand_landmarks.landmark[12]
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                
                cv2.circle(frame, (ix, iy), 8, (0, 255, 255), cv2.FILLED)
                cv2.circle(frame, (mx, my), 8, (255, 0, 255), cv2.FILLED)
        
        return frame, results
        
    def find_position(self, frame, hand_no=0, smooth=None):
        """
        Find the position of hand landmarks and optionally apply smoothing.
        
        Args:
            frame (numpy.ndarray): Input image frame.
            hand_no (int): Which hand to track if multiple are detected.
            smooth (bool, optional): Whether to apply smoothing to landmark positions.
            
        Returns:
            list: List of landmark positions [(x1,y1), (x2,y2), ...].
            bool: Whether a hand was detected.
        """
        # Use config value if not specified
        if smooth is None:
            smooth = config.LANDMARK_SMOOTHING
            
        frame_height, frame_width, _ = frame.shape
        self.landmark_list = []
        
        # Process the frame
        _, results = self.find_hands(frame, draw=False)
        
        if results.multi_hand_landmarks:
            # Get the specified hand
            if len(results.multi_hand_landmarks) > hand_no:
                hand = results.multi_hand_landmarks[hand_no]
                
                # Extract landmark coordinates
                for idx, landmark in enumerate(hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    self.landmark_list.append((x, y))
                
                # Apply smoothing if requested
                if smooth and self.landmark_list:
                    self.landmark_history.append(self.landmark_list.copy())
                    
                    # Keep history at the desired length
                    if len(self.landmark_history) > self.history_length:
                        self.landmark_history.pop(0)
                    
                    # Average the landmarks over history
                    if len(self.landmark_history) > 1:
                        smoothed_landmarks = []
                        for i in range(len(self.landmark_list)):
                            x_sum = sum(history[i][0] for history in self.landmark_history)
                            y_sum = sum(history[i][1] for history in self.landmark_history)
                            x_avg = int(x_sum / len(self.landmark_history))
                            y_avg = int(y_sum / len(self.landmark_history))
                            smoothed_landmarks.append((x_avg, y_avg))
                        return smoothed_landmarks, True
                
                return self.landmark_list, True
        
        # Reset history if no hand detected
        self.landmark_history = []
        return [], False
    
    def get_finger_states(self, hand_landmarks):
        """
        Determine which fingers are extended based on hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.
            
        Returns:
            list: Boolean list indicating which fingers are up [thumb, index, middle, ring, pinky].
        """
        if not hand_landmarks:
            return [False, False, False, False, False]
            
        # Get landmarks as a list for easier processing
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
        
        # Check thumb using angle 
        thumb_angle = self._angle_between_points(
            (landmarks[0][0], landmarks[0][1]),  # wrist
            (landmarks[2][0], landmarks[2][1]),  # thumb MCP
            (landmarks[4][0], landmarks[4][1])   # thumb tip
        )
        thumb_up = thumb_angle > 150
        
        # Check finger extensions by comparing y position of tips to PIPs
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_pips = [6, 10, 14, 18]  # Corresponding PIP joints
        
        fingers_up = []
        for tip, pip in zip(finger_tips, finger_pips):
            # A finger is up if the tip is higher (smaller y value) than the pip joint
            if landmarks[tip][1] < landmarks[pip][1]:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        
        return [thumb_up] + fingers_up
    
    def recognize_gesture(self, finger_states):
        """
        Recognize hand gesture based on finger states.
        
        Args:
            finger_states (list): Boolean list indicating which fingers are up 
                                   [thumb, index, middle, ring, pinky].
        
        Returns:
            str: The recognized gesture type.
        """
        if not finger_states or len(finger_states) != 5:
            return GestureType.UNKNOWN
        
        # Unpack finger states
        thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states
        
        # NEUTRAL: Open hand (all or most fingers up)
        if sum(finger_states) >= 4:
            return GestureType.NEUTRAL
            
        # SCROLL: Thumb, index and middle fingers up, others down
        elif thumb_up and index_up and middle_up and not ring_up and not pinky_up:
            return GestureType.SCROLL
            
        # CURSOR_MOVE: Peace sign (index and middle fingers up, others down)
        elif index_up and middle_up and not ring_up and not pinky_up:
            # Calculate distance between index and middle fingertips
            if hasattr(self, 'landmarks') and self.landmarks is not None:
                try:
                    # Access landmark points directly
                    index_tip = self.landmarks.landmark[8]
                    middle_tip = self.landmarks.landmark[12]
                    
                    # Get frame dimensions for converting normalized coordinates
                    h, w = 1, 1  # Default values if frame dimensions are not available
                    if hasattr(self, 'frame_shape'):
                        h, w, _ = self.frame_shape
                    
                    # Convert to pixel coordinates
                    ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                    mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                    
                    # Calculate Euclidean distance in pixel space
                    distance_px = np.sqrt((ix - mx)**2 + (iy - my)**2)
                    
                    # Calculate a normalized distance (as proportion of hand size)
                    wrist = self.landmarks.landmark[0]
                    middle_mcp = self.landmarks.landmark[9]
                    hand_width_dx = (wrist.x - middle_mcp.x) * w
                    hand_width_dy = (wrist.y - middle_mcp.y) * h
                    hand_width = np.sqrt(hand_width_dx**2 + hand_width_dy**2)
                    
                    # Threshold as a proportion of hand size
                    # Making the threshold larger (15% instead of 10%) for easier detection
                    threshold = max(30, hand_width * 0.15)  # At least 30 pixels or 15% of hand width
                    
                    # For debugging
                    # print(f"Distance: {distance_px:.1f}, Threshold: {threshold:.1f}, Hand width: {hand_width:.1f}")
                    
                    if distance_px < threshold:
                        return GestureType.DOUBLE_CLICK
                except (IndexError, AttributeError) as e:
                    # Handle potential errors gracefully
                    print(f"Error in finger distance calculation: {e}")
            
            return GestureType.CURSOR_MOVE
            
        # LEFT_CLICK: Middle finger up, index finger down (opposite of before)
        elif not index_up and middle_up and not ring_up and not pinky_up:
            return GestureType.LEFT_CLICK
            
        # RIGHT_CLICK: Index finger up, middle finger down (opposite of before)
        elif index_up and not middle_up and not ring_up and not pinky_up:
            return GestureType.RIGHT_CLICK
            
        # Unrecognized gesture
        else:
            return GestureType.UNKNOWN
    
    def get_stable_gesture(self, landmarks):
        """
        Get stable gesture over multiple frames to prevent flickering.
        
        Args:
            landmarks: MediaPipe hand landmarks.
            
        Returns:
            str: The stable gesture type.
        """
        if landmarks is None:
            self.gesture_history = []
            return GestureType.UNKNOWN
        
        finger_states = self.get_finger_states(landmarks)
        current_gesture = self.recognize_gesture(finger_states)
        
        # Add to history
        self.gesture_history.append(current_gesture)
        
        # Keep history at maximum length
        if len(self.gesture_history) > self.gesture_history_length:
            self.gesture_history.pop(0)
        
        # Return most common gesture in history
        if self.gesture_history:
            from collections import Counter
            return Counter(self.gesture_history).most_common(1)[0][0]
        else:
            return GestureType.UNKNOWN
    
    def _angle_between_points(self, p1, p2, p3):
        """
        Calculate the angle between three points.
        
        Args:
            p1, p2, p3: Three points [(x,y), (x,y), (x,y)]
        
        Returns:
            float: Angle in degrees
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        # Calculate the angle in radians
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        # Convert to degrees
        return np.degrees(angle)
    
    def get_gesture_info(self, landmarks):
        """
        Get comprehensive gesture information for the mouse controller.
        
        Args:
            landmarks: MediaPipe hand landmarks.
            
        Returns:
            dict: Gesture information including type, cursor position, etc.
        """
        if landmarks is None:
            return {
                "gesture": GestureType.UNKNOWN,
                "cursor_position": None,
                "is_clicking": False,
                "click_type": None,
                "is_scrolling": False
            }
        
        # Store landmarks for distance calculations in recognize_gesture
        self.landmarks = landmarks
        
        # Get stable gesture
        gesture = self.get_stable_gesture(landmarks)
        
        # Determine cursor position based on fingertips for CURSOR_MOVE
        cursor_position = None
        is_scrolling = False
        
        if gesture in [GestureType.CURSOR_MOVE, GestureType.DOUBLE_CLICK]:
            # Get average of index and middle finger tips for more stability
            index_tip = landmarks.landmark[8]
            middle_tip = landmarks.landmark[12]
            cursor_position = ((index_tip.x + middle_tip.x) / 2, (index_tip.y + middle_tip.y) / 2)
        elif gesture == GestureType.RIGHT_CLICK:
            # Use index finger tip for positioning
            index_tip = landmarks.landmark[8]
            cursor_position = (index_tip.x, index_tip.y)
        elif gesture == GestureType.LEFT_CLICK:
            # Use middle finger tip for positioning
            middle_tip = landmarks.landmark[12]
            cursor_position = (middle_tip.x, middle_tip.y)
        elif gesture == GestureType.SCROLL:
            # Use middle finger tip for scroll position
            middle_tip = landmarks.landmark[12]
            cursor_position = (middle_tip.x, middle_tip.y)
            is_scrolling = True
        
        # Determine click information
        is_clicking = gesture in [GestureType.LEFT_CLICK, GestureType.RIGHT_CLICK, GestureType.DOUBLE_CLICK]
        click_type = None
        if gesture == GestureType.LEFT_CLICK:
            click_type = "left"
        elif gesture == GestureType.RIGHT_CLICK:
            click_type = "right"
        elif gesture == GestureType.DOUBLE_CLICK:
            click_type = "double"
        
        return {
            "gesture": gesture,
            "cursor_position": cursor_position,
            "is_clicking": is_clicking,
            "click_type": click_type,
            "is_scrolling": is_scrolling
        } 