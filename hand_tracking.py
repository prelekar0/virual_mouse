import cv2
import mediapipe as mp
import numpy as np
import math
import config

class GestureType:
    NEUTRAL = "NEUTRAL"         # Open hand (all fingers up)
    CURSOR_MOVE = "CURSOR_MOVE" # Index and middle fingers up
    LEFT_CLICK = "LEFT_CLICK"   # Index finger down, middle finger up
    RIGHT_CLICK = "RIGHT_CLICK" # Index finger up, middle finger down
    DOUBLE_CLICK = "DOUBLE_CLICK" # Index and middle fingers close together
    SCROLL = "SCROLL"           # Three fingers up (thumb, index, middle)
    DRAG = "DRAG"  # Added drag gesture
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
        if frame is not None and len(frame.shape) >= 2:
            self.frame_shape = frame.shape
        else:
            # Default shape if frame is invalid
            self.frame_shape = (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3)
        
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
                
                # Draw additional information for better UX
                h, w, c = self.frame_shape
                
                # Get gesture information
                gesture_info = self.get_gesture_info(hand_landmarks)
                gesture = gesture_info["gesture"]
                
                # Add visual indication of gesture
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
    
    def _calculate_distance(self, landmark1, landmark2):
        """
        Calculate the Euclidean distance between two landmarks in normalized coordinates.
        
        Args:
            landmark1: First landmark point
            landmark2: Second landmark point
            
        Returns:
            float: Euclidean distance between the landmarks
        """
        return math.sqrt(
            (landmark1.x - landmark2.x) ** 2 + 
            (landmark1.y - landmark2.y) ** 2
        )
    
    def get_finger_states(self, hand_landmarks):
        """
        Determine which fingers are extended based on hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks.
            
        Returns:
            list: Boolean list indicating which fingers are up [thumb, index, middle, ring, pinky].
        """
        if hand_landmarks is None:
            return None
        
        # Get all landmarks
        points = []
        for landmark in hand_landmarks.landmark:
            points.append((landmark.x, landmark.y))
        
        # Define landmark indices
        tip_ids = [4, 8, 12, 16, 20]  # Fingertips
        pip_ids = [3, 7, 11, 15, 19]  # Finger PIP joints (middle joints)
        mcp_ids = [2, 6, 10, 14, 18]  # Finger MCP joints (knuckles)
        wrist_id = 0
        
        # Determine thumb state
        # For thumb, compare with first knuckle/MCP
        thumb_tip = points[tip_ids[0]]
        thumb_mcp = points[mcp_ids[0]]
        wrist = points[wrist_id]
        
        # Check if thumb is to the side (x-position)
        # Different method based on which hand it is (left vs right)
        # Check if thumb is significantly to the side of the index MCP
        index_mcp = points[mcp_ids[1]]
        thumb_side_threshold = 0.05
        
        # Thumb is up if the x value is significantly different from the index MCP
        thumb_up = abs(thumb_tip[0] - index_mcp[0]) > thumb_side_threshold
        
        # Check other fingers (index, middle, ring, pinky)
        # A finger is up if the tip y-position is higher (smaller value) than the PIP joint
        fingers_up = []
        for i in range(1, 5):
            finger_tip_y = points[tip_ids[i]][1]
            finger_pip_y = points[pip_ids[i]][1]
            finger_mcp_y = points[mcp_ids[i]][1]
            
            # The finger is considered up if the tip is higher than both joints
            # Using a slightly more generous threshold for better recognition
            threshold = 0.02  # Small threshold to account for camera angle
            finger_up = finger_tip_y < finger_pip_y - threshold
            fingers_up.append(finger_up)
        
        return [thumb_up] + fingers_up
    
    def recognize_gesture(self, landmarks=None, finger_states=None):
        """
        Recognize the current gesture based on finger states.
        
        Args:
            landmarks: Optional hand landmarks if available
            finger_states: List of booleans indicating which fingers are up
                           [thumb, index, middle, ring, pinky]
            
        Returns:
            str: Recognized gesture type from GestureType class
        """
        # Check if landmarks exist
        if landmarks is None and finger_states is None:
            return GestureType.UNKNOWN
            
        # Get finger states if not provided
        if finger_states is None and landmarks is not None:
            finger_states = self.get_finger_states(landmarks)
        
        # Still no finger states, return unknown
        if finger_states is None or len(finger_states) != 5:
            return GestureType.UNKNOWN
        
        # Unpack finger states
        thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states
        
        # Handle DRAG gesture with improved detection
        # More stable and accurate pinch detection
        if landmarks is not None:
            thumb_tip = landmarks.landmark[4]  # Thumb tip
            index_tip = landmarks.landmark[8]  # Index tip
            thumb_ip = landmarks.landmark[3]   # Thumb IP joint
            index_pip = landmarks.landmark[6]  # Index PIP joint
            
            # Calculate distances for better pinch detection
            thumb_to_index_distance = self._calculate_distance(thumb_tip, index_tip)
            
            # Use a combination of finger states and pinch distance for more accurate detection
            # Check for pinch - thumb tip close to index tip
            if thumb_to_index_distance < 0.07:  # Increased threshold for better detection
                # For drag, we want other fingers to be more extended (not in a fist)
                if middle_up and ring_up and pinky_up:
                    # Check if thumb is in a pinching position (not just up)
                    thumb_extended = thumb_tip.y < thumb_ip.y
                    index_bent = not (index_tip.y < index_pip.y)
                    
                    if thumb_extended and index_bent:
                        # This is a clear pinch-and-drag gesture
                        return GestureType.DRAG
        
        # LEFT_CLICK from neutral - Only index finger down, others up
        if not index_up and thumb_up and middle_up and ring_up and pinky_up:
            return GestureType.LEFT_CLICK
        
        # RIGHT_CLICK from neutral - Only middle finger down, others up
        if index_up and not middle_up and thumb_up and ring_up and pinky_up:
            return GestureType.RIGHT_CLICK
        
        # NEUTRAL - Open hand (all fingers up)
        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            return GestureType.NEUTRAL
        
        # NEUTRAL - At least 4 fingers up also counts as neutral
        if sum(finger_states) >= 4:
            return GestureType.NEUTRAL
        
        # SCROLL - Three fingers up (thumb, index, and middle) with ring and pinky down
        # Check scroll before cursor move to give it priority
        if thumb_up and index_up and middle_up and not ring_up and not pinky_up:
            return GestureType.SCROLL
        
        # CURSOR_MOVE - Peace sign (index and middle fingers up, others down)
        if index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            # Calculate distance between index and middle fingertips for DOUBLE_CLICK detection
            if landmarks is not None:
                index_tip = landmarks.landmark[8]  # Index tip
                middle_tip = landmarks.landmark[12]  # Middle tip
                distance = self._calculate_distance(index_tip, middle_tip)
                
                # If fingers are very close, consider it a double click gesture
                double_click_threshold = 0.05
                if distance < double_click_threshold:
                    return GestureType.DOUBLE_CLICK
            
            return GestureType.CURSOR_MOVE
        
        # Standard LEFT_CLICK - Index finger down, middle finger up
        if not index_up and middle_up:
            return GestureType.LEFT_CLICK
        
        # Standard RIGHT_CLICK - Index finger up, middle finger down
        if index_up and not middle_up:
            return GestureType.RIGHT_CLICK
        
        # Fallback CURSOR_MOVE - If index and middle are up
        # This makes peace sign detection more forgiving
        if index_up and middle_up:
            return GestureType.CURSOR_MOVE
        
        # Default case if no gesture is recognized
        return GestureType.UNKNOWN
    
    def get_stable_gesture(self, landmarks):
        """
        Get a stable gesture over multiple frames to prevent flickering.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            str: The stable gesture type
        """
        # Handle None landmarks
        if landmarks is None:
            self.gesture_history = []
            return GestureType.UNKNOWN
            
        # Get the current gesture
        current_gesture = self.recognize_gesture(landmarks=landmarks)
        
        # Add to history
        self.gesture_history.append(current_gesture)
        
        # Keep history at the desired length
        if len(self.gesture_history) > self.gesture_history_length:
            self.gesture_history.pop(0)
        
        # Count occurrence of each gesture in history
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture in gesture_counts:
                gesture_counts[gesture] += 1
            else:
                gesture_counts[gesture] = 1
                
        # Find the most common gesture
        max_count = 0
        most_common_gesture = GestureType.UNKNOWN
        
        for gesture, count in gesture_counts.items():
            if count > max_count:
                max_count = count
                most_common_gesture = gesture
                
        # Only return the gesture if it's stable enough (appears in majority of frames)
        stability_threshold = self.gesture_history_length // 2
        if max_count > stability_threshold:
            return most_common_gesture
        
        # For LEFT_CLICK, be more strict to avoid accidental clicks
        if most_common_gesture == GestureType.LEFT_CLICK:
            # Require more frames for left click to be stable
            click_threshold = int(self.gesture_history_length * 0.7)
            if max_count >= click_threshold:
                return GestureType.LEFT_CLICK
            else:
                # If not stable enough, return the previous stable gesture
                # or NEUTRAL as default if no gesture was recognized yet
                prev_gestures = [g for g in self.gesture_history[:-1] 
                                if g != GestureType.LEFT_CLICK]
                if prev_gestures:
                    return max(set(prev_gestures), key=prev_gestures.count)
                return GestureType.NEUTRAL
                
        return most_common_gesture
    
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
        Get comprehensive information about the detected gesture.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            dict: Dictionary containing gesture type and additional info
        """
        # Default values
        info = {
            "gesture": GestureType.UNKNOWN,
            "cursor_position": None,
            "scroll_direction": None,
            "distance_factor": 1.0  # New field to indicate hand distance from camera
        }
        
        # Basic information - landmark positions
        h, w, c = self.frame_shape
        
        # Calculate distance factor based on hand size in frame
        # Get wrist and middle finger tip landmarks
        wrist = landmarks.landmark[0]
        middle_tip = landmarks.landmark[12]
        
        # Calculate distance between wrist and middle finger tip in pixels
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
        tip_x, tip_y = int(middle_tip.x * w), int(middle_tip.y * h)
        hand_length = math.sqrt((tip_x - wrist_x)**2 + (tip_y - wrist_y)**2)
        
        # Normalize by frame diagonal for consistent scaling across different resolutions
        frame_diagonal = math.sqrt(w**2 + h**2)
        normalized_hand_length = hand_length / frame_diagonal
        
        # Use hand length as a proxy for distance - larger hand means closer to camera
        # Typical values range from 0.05 (far) to 0.2 (close)
        # Map this to a distance factor between 0.7 and 1.3
        distance_factor = max(0.7, min(1.3, normalized_hand_length * 5))
        info["distance_factor"] = distance_factor
        
        # Get finger states
        finger_states = self.get_finger_states(landmarks)
        
        # Get stable gesture to reduce false detections
        gesture = self.get_stable_gesture(landmarks)
        info["gesture"] = gesture
        
        # Get cursor position based on index finger
        index_finger_tip = landmarks.landmark[8]
        index_x, index_y = index_finger_tip.x, index_finger_tip.y
        
        # Apply adaptive scaling based on distance factor
        # When hand is further away (smaller), we want larger movements to have same effect
        margin = int(config.MARGIN / distance_factor)
        
        # Store the cursor position
        info["cursor_position"] = (index_x, index_y)
        info["margin"] = margin
        
        return info 
    
    def recognize_hand_gesture(self, landmarks):
        """
        Recognize hand gesture and return both gesture type and info.
        
        Args:
            landmarks: Hand landmarks from MediaPipe
            
        Returns:
            tuple: (gesture_type, gesture_info)
        """
        # Get gesture info with cursor position and distance factor
        gesture_info = self.get_gesture_info(landmarks)
        
        # Get the recognized gesture
        gesture = gesture_info["gesture"]
        
        return gesture, gesture_info 