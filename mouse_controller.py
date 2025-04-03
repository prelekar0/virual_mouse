import pyautogui
import time
import numpy as np

class MouseController:
    """
    Class to handle mouse control actions with simplified methods.
    """
    
    def __init__(self, smoothing_factor=0.5, click_cooldown=0.5):
        """
        Initialize the mouse controller.
        
        Args:
            smoothing_factor (float): Factor for movement smoothing (1-20).
                                     Higher values = smoother but slower.
            click_cooldown (float): Time in seconds to wait between clicks.
        """
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Prevent mouse from triggering failsafe
        pyautogui.FAILSAFE = False
        
        # Movement smoothing
        self.smoothing_factor = smoothing_factor
        self.prev_x, self.prev_y = 0, 0
        
        # Click control
        self.click_cooldown = click_cooldown
        self.last_click_time = 0
        
        # Dynamic speed control
        self.prev_hand_pos = None
        self.prev_time = time.time()
        self.min_speed_multiplier = 0.5
        self.max_speed_multiplier = 3.0
        
        # Scrolling
        self.scroll_cooldown = 0.05  # seconds between scroll actions
        self.last_scroll_time = 0
        self.scroll_speed = 2  # base scroll speed
        self.prev_scroll_y = None
        self.prev_scroll_x = None
        
        print(f"Mouse Controller initialized with screen dimensions: {self.screen_width}x{self.screen_height}")
        print(f"Smoothing factor: {smoothing_factor}, Click cooldown: {click_cooldown}s")
    
    def smooth_move(self, x, y):
        """
        Move the mouse cursor with smoothing and dynamic speed based on hand movement.
        
        Args:
            x (float): Normalized x coordinate (0-1).
            y (float): Normalized y coordinate (0-1).
            
        Returns:
            tuple: The actual (x, y) position after smoothing.
        """
        # Calculate speed multiplier based on hand movement speed
        speed_multiplier = 1.0
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt > 0 and self.prev_hand_pos is not None:
            # Calculate hand movement speed in normalized coordinates per second
            dx = x - self.prev_hand_pos[0]
            dy = y - self.prev_hand_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)
            speed = distance / dt
            
            # Map speed to a multiplier (faster hand movement = faster cursor)
            # Typical hand movement speed is 0.05-0.5 units per second
            if speed > 0.01:  # Apply dynamic speed only if actually moving
                speed_multiplier = np.clip(
                    self.min_speed_multiplier + speed * 5,  # Scale factor of 5 for sensitivity
                    self.min_speed_multiplier,
                    self.max_speed_multiplier
                )
        
        # Update previous position and time
        self.prev_hand_pos = (x, y)
        self.prev_time = current_time
        
        # Convert normalized coordinates to screen coordinates
        target_x = int(x * self.screen_width)
        target_y = int(y * self.screen_height)
        
        # Apply exponential moving average for smoothing with dynamic speed
        if self.prev_x == 0 and self.prev_y == 0:
            # First movement, set directly
            smoothed_x = target_x
            smoothed_y = target_y
        else:
            # Apply smoothing with dynamic factor
            dynamic_smoothing = max(1.0, self.smoothing_factor / speed_multiplier)
            smoothed_x = self.prev_x + (target_x - self.prev_x) / dynamic_smoothing
            smoothed_y = self.prev_y + (target_y - self.prev_y) / dynamic_smoothing
        
        # Ensure coordinates are within screen boundaries
        smoothed_x = max(0, min(smoothed_x, self.screen_width - 1))
        smoothed_y = max(0, min(smoothed_y, self.screen_height - 1))
        
        # Update previous positions
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        
        try:
            # Move mouse - convert to integers to avoid potential float issues
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
        except Exception as e:
            print(f"Mouse movement error: {e}")
        
        return smoothed_x, smoothed_y
    
    def scroll_vertical(self, direction, amount=None):
        """
        Perform vertical scrolling.
        
        Args:
            direction (str): "up" or "down"
            amount (float, optional): Scrolling amount, calculated automatically if None
            
        Returns:
            bool: True if scroll was performed, False if on cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return False
        
        # Set scroll amount
        if amount is None:
            amount = self.scroll_speed
        
        # Scroll up or down
        try:
            clicks = int(amount)
            if direction == "up":
                pyautogui.scroll(clicks)  # Positive for up
            else:
                pyautogui.scroll(-clicks)  # Negative for down
                
            self.last_scroll_time = current_time
            return True
        except Exception as e:
            print(f"Scroll error: {e}")
            return False
    
    def scroll_horizontal(self, direction, amount=None):
        """
        Perform horizontal scrolling.
        
        Args:
            direction (str): "left" or "right"
            amount (float, optional): Scrolling amount, calculated automatically if None
            
        Returns:
            bool: True if scroll was performed, False if on cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scroll_time < self.scroll_cooldown:
            return False
        
        # Set scroll amount
        if amount is None:
            amount = self.scroll_speed
        
        # Scroll left or right
        try:
            clicks = int(amount)
            if direction == "left":
                pyautogui.hscroll(-clicks)  # Negative for left
            else:
                pyautogui.hscroll(clicks)   # Positive for right
                
            self.last_scroll_time = current_time
            return True
        except Exception as e:
            print(f"Horizontal scroll error: {e}")
            return False
    
    def handle_scroll(self, x_position, y_position):
        """
        Handle both vertical and horizontal scrolling based on hand position.
        
        Args:
            x_position (float): Normalized x-coordinate (0-1)
            y_position (float): Normalized y-coordinate (0-1)
            
        Returns:
            tuple: (scrolled, direction)
        """
        # Initialize previous positions if needed
        if self.prev_scroll_y is None:
            self.prev_scroll_y = y_position
            self.prev_scroll_x = x_position
            return False, None
        
        # Calculate changes in position
        delta_y = y_position - self.prev_scroll_y
        delta_x = x_position - self.prev_scroll_x
        
        # Define thresholds for movement
        threshold = 0.01
        
        # Determine if this is primarily a horizontal or vertical scroll
        # by comparing which delta is larger
        if abs(delta_x) > abs(delta_y) and abs(delta_x) > threshold:
            # Horizontal scrolling has priority
            amount = min(10, max(1, abs(delta_x) * 100))
            
            if delta_x < 0:
                # Moving hand left = scroll left
                if self.scroll_horizontal("left", amount):
                    self.prev_scroll_x = x_position
                    self.prev_scroll_y = y_position
                    return True, "left"
            else:
                # Moving hand right = scroll right
                if self.scroll_horizontal("right", amount):
                    self.prev_scroll_x = x_position
                    self.prev_scroll_y = y_position
                    return True, "right"
        
        # If not horizontal or below horizontal threshold, check vertical
        elif abs(delta_y) > threshold:
            # Calculate scroll amount based on movement size
            amount = min(10, max(1, abs(delta_y) * 100))
            
            if delta_y < 0:
                # Moving hand up = scroll up
                if self.scroll_vertical("up", amount):
                    self.prev_scroll_x = x_position
                    self.prev_scroll_y = y_position
                    return True, "up"
            else:
                # Moving hand down = scroll down
                if self.scroll_vertical("down", amount):
                    self.prev_scroll_x = x_position
                    self.prev_scroll_y = y_position
                    return True, "down"
        
        return False, None
    
    def left_click(self):
        """
        Perform a left mouse click with cooldown.
        
        Returns:
            bool: True if click was performed, False otherwise.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_click_time > self.click_cooldown:
            try:
                pyautogui.click()
                self.last_click_time = current_time
                return True
            except Exception as e:
                print(f"Left click error: {e}")
        
        return False
    
    def right_click(self):
        """
        Perform a right mouse click with cooldown.
        
        Returns:
            bool: True if click was performed, False otherwise.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_click_time > self.click_cooldown:
            try:
                pyautogui.rightClick()
                self.last_click_time = current_time
                return True
            except Exception as e:
                print(f"Right click error: {e}")
        
        return False
    
    def double_click(self):
        """
        Perform a double-click action.
        
        Returns:
            bool: True if double-click was performed, False otherwise.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_click_time > self.click_cooldown:
            try:
                print("Performing double-click")
                # Use direct mouse events for more reliable double-click
                pyautogui.click(clicks=2)
                self.last_click_time = current_time
                return True
            except Exception as e:
                print(f"Double click error: {e}")
        
        return False
    
    def start_drag(self, x, y):
        """
        Start a drag operation.
        
        Args:
            x (float): X coordinate to start dragging from.
            y (float): Y coordinate to start dragging from.
        """
        if not self.is_dragging:
            try:
                self.is_dragging = True
                self.drag_start_pos = (x, y)
                
                # Ensure coordinates are within screen boundaries
                x = max(0, min(x, self.screen_width - 1))
                y = max(0, min(y, self.screen_height - 1))
                
                pyautogui.mouseDown(int(x), int(y), button="left")
            except Exception as e:
                print(f"Start drag error: {e}")
                self.is_dragging = False
    
    def continue_drag(self, x, y):
        """
        Continue a drag operation.
        
        Args:
            x (float): X coordinate to drag to.
            y (float): Y coordinate to drag to.
        """
        if self.is_dragging:
            try:
                # Ensure coordinates are within screen boundaries
                x = max(0, min(x, self.screen_width - 1))
                y = max(0, min(y, self.screen_height - 1))
                
                pyautogui.moveTo(int(x), int(y))
            except Exception as e:
                print(f"Continue drag error: {e}")
    
    def stop_drag(self, x, y):
        """
        End a drag operation.
        
        Args:
            x (float): X coordinate to end dragging at.
            y (float): Y coordinate to end dragging at.
        """
        if self.is_dragging:
            try:
                # Ensure coordinates are within screen boundaries
                x = max(0, min(x, self.screen_width - 1))
                y = max(0, min(y, self.screen_height - 1))
                
                pyautogui.mouseUp(int(x), int(y), button="left")
            except Exception as e:
                print(f"Stop drag error: {e}")
            finally:
                self.is_dragging = False
                self.drag_start_pos = None
    
    def map_coordinates(self, x, y, input_width, input_height, margin=None):
        """
        Map input coordinates to screen coordinates.
        
        Args:
            x (float): Input x coordinate.
            y (float): Input y coordinate.
            input_width (int): Width of input space.
            input_height (int): Height of input space.
            margin (int, optional): Margin from edges in input space.
            
        Returns:
            tuple: Mapped (x, y) coordinates for screen.
        """
        # Use config margin if not specified
        if margin is None:
            margin = config.MARGIN
            
        # Apply margins
        in_min_x, in_max_x = margin, input_width - margin
        in_min_y, in_max_y = margin, input_height - margin
        
        # Map to screen coordinates - use simple linear mapping
        screen_x = max(0, min(self.screen_width - 1, 
                             self._map_range(x, in_min_x, in_max_x, 0, self.screen_width)))
        screen_y = max(0, min(self.screen_height - 1, 
                             self._map_range(y, in_min_y, in_max_y, 0, self.screen_height)))
        
        return screen_x, screen_y
    
    def _map_range(self, value, in_min, in_max, out_min, out_max):
        """
        Map a value from one range to another.
        
        Args:
            value (float): Input value.
            in_min (float): Input range minimum.
            in_max (float): Input range maximum.
            out_min (float): Output range minimum.
            out_max (float): Output range maximum.
            
        Returns:
            float: Mapped value.
        """
        # Check for division by zero
        if in_max == in_min:
            return out_min
        
        # Linear interpolation
        mapped_value = out_min + (((value - in_min) / (in_max - in_min)) * (out_max - out_min))
        return mapped_value 