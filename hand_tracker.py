#!/usr/bin/env python3
import cv2
import numpy as np
from enum import Enum
from collections import Counter, deque

class Gesture(Enum):
    NONE = 0
    FIST = 1          # Left click
    TWO_FINGERS = 2   # Right click
    OPEN_PALM = 3     # Scroll mode
    POINTING = 4      # Default cursor movement

class HandTracker:
    def __init__(self, 
                 static_image_mode=False, 
                 max_num_hands=1, 
                 min_detection_confidence=0.7, 
                 min_tracking_confidence=0.7,
                 history_length=5):
        """
        Initialize the hand tracker with OpenCV's features.
        
        Args:
            static_image_mode: Not used in this implementation
            max_num_hands: Not used in this implementation
            min_detection_confidence: Used for threshold values
            min_tracking_confidence: Not used in this implementation
            history_length: Number of frames to keep for gesture stability
        """
        # Initialize parameters
        self.min_detection_confidence = min_detection_confidence
        self.history_length = history_length
        self.gesture_history = deque(maxlen=history_length)
        self.position_history = deque(maxlen=history_length * 2)
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=60,         # Shorter history for faster adaptation
            varThreshold=20,    # Lower threshold for more sensitive detection
            detectShadows=False
        )
        
        # Parameters for hand detection - modified for better sensitivity
        self.hand_area_min = 3000   # Minimum area to detect a hand
        self.hand_area_max = 100000  # Maximum area for a hand
        
        # Skin color detection parameters (in HSV space)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Init hand landmark positions
        self.landmarks = None
        self.prev_landmarks = None
        
        # Init hand contour
        self.hand_contour = None
        self.prev_contour = None
        
        # Frame counter
        self.frame_count = 0
        
        # Debug visualization
        self.debug_img = None
        
        # Region of interest for tracking
        self.roi = None
        self.roi_established = False
        
    def process_frame(self, frame):
        """
        Process a video frame and detect hands.
        
        Args:
            frame: Input BGR image
            
        Returns:
            results: Hand detection results object with multi_hand_landmarks attribute
        """
        self.frame_count += 1
        
        # Make a copy for debugging
        debug_img = frame.copy()
        
        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Skin color mask
        skin_mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Apply morphological operations to clean up the skin mask
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)
        
        # Display the skin mask in a corner for debugging
        h, w = frame.shape[:2]
        small_skin_mask = cv2.resize(skin_mask, (w//4, h//4))
        debug_img[0:h//4, 0:w//4] = cv2.cvtColor(small_skin_mask, cv2.COLOR_GRAY2BGR)
        
        # Apply background subtraction to isolate moving objects
        # Convert to grayscale for background subtraction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Combine skin detection with motion detection
        # Only consider skin-colored areas that are also moving
        combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
        
        # Clean up the combined mask
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
        
        # Display combined mask
        small_combined = cv2.resize(combined_mask, (w//4, h//4))
        debug_img[0:h//4, w//4:w//2] = cv2.cvtColor(small_combined, cv2.COLOR_GRAY2BGR)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw all contours on debug image
        cv2.drawContours(debug_img, contours, -1, (0, 255, 255), 1)
        
        # Find the largest contour that could be a hand
        hand_contour = None
        max_area = self.hand_area_min
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Check if the contour is within the size range for a hand
            if self.hand_area_min < area < self.hand_area_max:
                # Calculate the solidity (area / convex hull area) to filter out non-hand shapes
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Hands typically have solidity between 0.6 and 0.9
                if 0.4 < solidity < 0.95:
                    if area > max_area:
                        max_area = area
                        hand_contour = contour
        
        self.hand_contour = hand_contour
        
        # Extract key points from the hand contour
        landmarks = []
        
        if hand_contour is not None:
            # Draw the main hand contour in a different color
            cv2.drawContours(debug_img, [hand_contour], 0, (0, 255, 0), 2)
            
            # Get bounding rectangle and draw it
            x, y, w, h = cv2.boundingRect(hand_contour)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Update ROI for future frames
            self.roi = (x, y, w, h)
            self.roi_established = True
            
            # Find the convex hull
            try:
                # First try using returnPoints=False for convexity defects
                hull = cv2.convexHull(hand_contour, returnPoints=False)
                
                # Draw the convex hull
                hull_points = cv2.convexHull(hand_contour, returnPoints=True)
                cv2.drawContours(debug_img, [hull_points], 0, (255, 0, 0), 2)
                
                # Get defects (spaces between fingers)
                if len(hull) > 3:
                    defects = cv2.convexityDefects(hand_contour, hull)
                    
                    if defects is not None and len(defects) > 0:
                        # Centroid of the contour
                        M = cv2.moments(hand_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = 0, 0
                        
                        # Draw palm center
                        cv2.circle(debug_img, (cx, cy), 8, (0, 0, 255), -1)
                        
                        # Add palm center as landmark 0
                        landmarks.append((cx, cy))
                        
                        # Extract and filter finger tips
                        finger_tips = []
                        
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            start = tuple(hand_contour[s][0])
                            end = tuple(hand_contour[e][0])
                            far = tuple(hand_contour[f][0])
                            
                            # Calculate the distance from defect point to palm center
                            # to filter out false defects
                            dist_to_center = np.sqrt((far[0] - cx)**2 + (far[1] - cy)**2)
                            
                            # Only consider defects that are close to the palm center
                            if dist_to_center < w/2:
                                # Draw defect points
                                cv2.circle(debug_img, far, 5, (255, 0, 255), -1)
                                
                                # Draw lines from defect point to convex hull points
                                cv2.line(debug_img, start, far, (0, 255, 255), 2)
                                cv2.line(debug_img, end, far, (0, 255, 255), 2)
                                
                                # Add finger tips
                                finger_tips.append(start)
                                finger_tips.append(end)
                        
                        # Check if the point is likely to be a finger tip
                        # by measuring its distance from the centroid
                        # and making sure it's above the palm center
                        filtered_tips = []
                        for pt in finger_tips:
                            dist_to_center = np.sqrt((pt[0] - cx)**2 + (pt[1] - cy)**2)
                            # Only consider points that are far enough from the center
                            # and above the palm center (y coordinate is smaller)
                            if dist_to_center > w/4 and pt[1] < cy:
                                filtered_tips.append(pt)
                        
                        # Filter unique finger tips (we may have duplicates)
                        unique_tips = []
                        for pt in filtered_tips:
                            # Check if point is far enough from all existing points
                            is_unique = True
                            for existing in unique_tips:
                                dist = np.sqrt((pt[0] - existing[0])**2 + (pt[1] - existing[1])**2)
                                if dist < 20:  # If points are close, consider them the same
                                    is_unique = False
                                    break
                            
                            if is_unique:
                                unique_tips.append(pt)
                                cv2.circle(debug_img, pt, 10, (0, 255, 0), -1)
                        
                        # Add unique finger tips to landmarks
                        landmarks.extend(unique_tips[:4])  # Take up to 4 tips
            except:
                # If convexity defects approach fails, try extrema points
                pass
            
            # If we couldn't extract enough landmarks, use extrema points
            if len(landmarks) < 5:
                # Find extrema points
                leftmost = tuple(hand_contour[hand_contour[:,:,0].argmin()][0])
                rightmost = tuple(hand_contour[hand_contour[:,:,0].argmax()][0])
                topmost = tuple(hand_contour[hand_contour[:,:,1].argmin()][0])
                bottommost = tuple(hand_contour[hand_contour[:,:,1].argmax()][0])
                
                # Draw extrema points
                cv2.circle(debug_img, leftmost, 8, (255, 0, 0), -1)
                cv2.circle(debug_img, rightmost, 8, (0, 255, 0), -1)
                cv2.circle(debug_img, topmost, 8, (0, 0, 255), -1)
                cv2.circle(debug_img, bottommost, 8, (255, 255, 0), -1)
                
                # Get centroid
                M = cv2.moments(hand_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                
                # Draw centroid
                cv2.circle(debug_img, (cx, cy), 8, (0, 0, 255), -1)
                
                # Use these points as basic landmarks
                landmarks = [(cx, cy), topmost, rightmost, bottommost, leftmost]
        
        self.prev_landmarks = self.landmarks
        self.landmarks = landmarks
        self.debug_img = debug_img
        
        # Create a class to mimic mediapipe's results structure
        class Results:
            def __init__(self, landmarks, frame_shape, debug_img=None):
                self.multi_hand_landmarks = []
                self.debug_img = debug_img
                if landmarks and len(landmarks) >= 5:
                    self.multi_hand_landmarks.append(self._convert_to_landmarks(landmarks, frame_shape))
            
            def _convert_to_landmarks(self, points, frame_shape):
                class HandLandmark:
                    def __init__(self, landmarks):
                        self.landmark = landmarks
                
                # Create normalized landmarks (21 points to match mediapipe format)
                normalized_landmarks = []
                h, w = frame_shape[:2]
                
                # We'll use the limited number of points we have and generate synthetic landmarks
                # to match the mediapipe format
                for i in range(21):
                    normalized_landmarks.append(None)
                
                # The index below is approximate mapping to mediapipe hand landmarks
                if len(points) >= 5:
                    # 0: Palm/wrist
                    normalized_landmarks[0] = self._normalize_point(points[0], w, h)
                    
                    # 4: Thumb tip
                    normalized_landmarks[4] = self._normalize_point(points[1], w, h)
                    
                    # 8: Index finger tip
                    normalized_landmarks[8] = self._normalize_point(points[2], w, h)
                    
                    # 12: Middle finger tip
                    normalized_landmarks[12] = self._normalize_point(points[3], w, h)
                    
                    # 16: Ring finger tip (if available)
                    normalized_landmarks[16] = self._normalize_point(points[4], w, h)
                    
                    # Generate other landmarks by interpolation
                    # For the thumb
                    normalized_landmarks[1] = self._interpolate(normalized_landmarks[0], normalized_landmarks[4], 0.25)
                    normalized_landmarks[2] = self._interpolate(normalized_landmarks[0], normalized_landmarks[4], 0.5)
                    normalized_landmarks[3] = self._interpolate(normalized_landmarks[0], normalized_landmarks[4], 0.75)
                    
                    # For index finger
                    normalized_landmarks[5] = self._interpolate(normalized_landmarks[0], normalized_landmarks[8], 0.25)
                    normalized_landmarks[6] = self._interpolate(normalized_landmarks[0], normalized_landmarks[8], 0.5)
                    normalized_landmarks[7] = self._interpolate(normalized_landmarks[0], normalized_landmarks[8], 0.75)
                    
                    # For middle finger
                    normalized_landmarks[9] = self._interpolate(normalized_landmarks[0], normalized_landmarks[12], 0.25)
                    normalized_landmarks[10] = self._interpolate(normalized_landmarks[0], normalized_landmarks[12], 0.5)
                    normalized_landmarks[11] = self._interpolate(normalized_landmarks[0], normalized_landmarks[12], 0.75)
                    
                    # For ring finger
                    normalized_landmarks[13] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 0.25)
                    normalized_landmarks[14] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 0.5)
                    normalized_landmarks[15] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 0.75)
                    
                    # For pinky (generate from palm center and ring finger)
                    normalized_landmarks[17] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 0.25, offset=0.1)
                    normalized_landmarks[18] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 0.5, offset=0.1)
                    normalized_landmarks[19] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 0.75, offset=0.1)
                    normalized_landmarks[20] = self._interpolate(normalized_landmarks[0], normalized_landmarks[16], 1.0, offset=0.2)
                
                return HandLandmark(normalized_landmarks)
            
            def _normalize_point(self, point, width, height):
                """Convert pixel coordinates to normalized coordinates"""
                class Landmark:
                    def __init__(self, x, y, z=0):
                        self.x = x
                        self.y = y
                        self.z = z
                
                if point is None:
                    return Landmark(0.5, 0.5, 0)
                
                return Landmark(point[0] / width, point[1] / height, 0)
            
            def _interpolate(self, landmark1, landmark2, ratio, offset=0.0):
                """Interpolate between two landmarks"""
                class Landmark:
                    def __init__(self, x, y, z=0):
                        self.x = x
                        self.y = y
                        self.z = z
                
                if landmark1 is None or landmark2 is None:
                    return Landmark(0.5, 0.5, 0)
                
                x = landmark1.x + (landmark2.x - landmark1.x) * ratio
                y = landmark1.y + (landmark2.y - landmark1.y) * ratio
                
                # Apply offset if needed
                if offset != 0.0:
                    # Perpendicular direction
                    dx = landmark2.y - landmark1.y
                    dy = -(landmark2.x - landmark1.x)
                    
                    # Normalize
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        dx /= length
                        dy /= length
                    
                    # Apply offset
                    x += dx * offset
                    y += dy * offset
                
                return Landmark(x, y, 0)
        
        return Results(landmarks, frame.shape, debug_img)
    
    def draw_landmarks(self, image, hand_landmarks):
        """
        Draw hand landmarks on the image.
        
        Args:
            image: Input image (BGR)
            hand_landmarks: Hand landmarks
            
        Returns:
            image: Image with drawn landmarks
        """
        # If we have debug image, use it
        if self.debug_img is not None:
            # Make a copy to avoid modifying the original debug image
            image = self.debug_img.copy()
        
        # Draw the hand contour if available
        if self.hand_contour is not None:
            cv2.drawContours(image, [self.hand_contour], 0, (0, 255, 0), 2)
        
        # Draw the landmarks
        for i in range(21):
            landmark = hand_landmarks.landmark[i]
            if landmark is not None:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        # Draw connections between landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (13, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (0, 17), (0, 5), (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            
            start_landmark = hand_landmarks.landmark[start_idx]
            end_landmark = hand_landmarks.landmark[end_idx]
            
            if start_landmark is not None and end_landmark is not None:
                start_x = int(start_landmark.x * image.shape[1])
                start_y = int(start_landmark.y * image.shape[0])
                end_x = int(end_landmark.x * image.shape[1])
                end_y = int(end_landmark.y * image.shape[0])
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Add text about how to use
        cv2.putText(image, "Move hand slowly", (10, image.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, "Place hand against plain background", (10, image.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image
    
    def get_finger_states(self, hand_landmarks):
        """
        Determine which fingers are extended based on landmark positions.
        
        Args:
            hand_landmarks: Hand landmarks
            
        Returns:
            extended: List of booleans indicating whether each finger is extended
                      [thumb, index, middle, ring, pinky]
        """
        # Get the landmarks we need
        wrist = hand_landmarks.landmark[0]
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Get finger bases
        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]
        ring_base = hand_landmarks.landmark[13]
        pinky_base = hand_landmarks.landmark[17]
        
        # Check if each finger is extended
        extended = []
        
        # Thumb: Check if it's to the right of the wrist
        if wrist and thumb_tip:
            extended.append(thumb_tip.x > wrist.x)
        else:
            extended.append(False)
        
        # For other fingers, check if the tip is above the base
        finger_pairs = [
            (index_tip, index_base),
            (middle_tip, middle_base),
            (ring_tip, ring_base),
            (pinky_tip, pinky_base)
        ]
        
        for tip, base in finger_pairs:
            if tip is not None and base is not None:
                extended.append(tip.y < base.y)
            else:
                extended.append(False)
        
        return extended
    
    def detect_gesture(self, finger_states):
        """
        Identify the gesture based on extended fingers.
        
        Args:
            finger_states: List of booleans for each finger [thumb, index, middle, ring, pinky]
            
        Returns:
            gesture: Detected gesture from Gesture enum
        """
        # Fist - no fingers extended (or just thumb)
        if sum(finger_states[1:]) == 0:
            return Gesture.FIST
        
        # Two fingers - index and middle extended, others closed
        if finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]:
            return Gesture.TWO_FINGERS
        
        # Open palm - at least 3 fingers extended
        if sum(finger_states[1:]) >= 3:
            return Gesture.OPEN_PALM
        
        # Default - pointing or other gestures
        return Gesture.POINTING
    
    def get_stable_gesture(self, gesture):
        """
        Ensure the gesture is stable across multiple frames.
        
        Args:
            gesture: Current detected gesture
            
        Returns:
            stable_gesture: Most common gesture over the last few frames
        """
        self.gesture_history.append(gesture)
        
        # Return the most common gesture in the history
        if len(self.gesture_history) >= 3:  # Need at least 3 frames for stability
            gestures = list(self.gesture_history)
            return Counter(gestures).most_common(1)[0][0]
        
        return gesture
    
    def smooth_position(self, landmark, weight=0.5):
        """
        Apply smoothing to coordinates to reduce jitter.
        
        Args:
            landmark: Landmark containing x, y coordinates
            weight: Weight for current position vs. history (0-1)
            
        Returns:
            smoothed_x, smoothed_y: Smoothed coordinates
        """
        if landmark is None:
            return 0.5, 0.5
            
        # Add current position to history
        self.position_history.append((landmark.x, landmark.y))
        
        if len(self.position_history) < 3:
            return landmark.x, landmark.y
        
        # Calculate the average of the last few positions
        history_length = min(len(self.position_history), 5)
        avg_x = sum(item[0] for item in list(self.position_history)[-history_length:]) / history_length
        avg_y = sum(item[1] for item in list(self.position_history)[-history_length:]) / history_length
        
        # Apply weighted average
        smoothed_x = weight * landmark.x + (1 - weight) * avg_x
        smoothed_y = weight * landmark.y + (1 - weight) * avg_y
        
        return smoothed_x, smoothed_y
    
    def release(self):
        """Release resources"""
        pass 