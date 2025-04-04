import pyautogui
import time
import numpy as np
import config
import math

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
        
        # Apply screen coverage factor
        if hasattr(config, 'SCREEN_COVERAGE_FACTOR'):
            self.effective_width = self.screen_width * config.SCREEN_COVERAGE_FACTOR
            self.effective_height = self.screen_height * config.SCREEN_COVERAGE_FACTOR
            self.screen_offset_x = (self.screen_width - self.effective_width) / 2
            self.screen_offset_y = (self.screen_height - self.effective_height) / 2
        else:
            self.effective_width = self.screen_width
            self.effective_height = self.screen_height
            self.screen_offset_x = 0
            self.screen_offset_y = 0
        
        # Prevent mouse from triggering failsafe
        pyautogui.FAILSAFE = False
        
        # Movement smoothing
        self.smoothing_factor = smoothing_factor
        self.prev_x, self.prev_y = 0, 0
        
        # For normalized coordinate tracking
        self.prev_norm_x = None
        self.prev_norm_y = None
        
        # Set margin (pixels from screen edge to avoid)
        self.margin = config.MARGIN
        self.margin_factor = self.margin / min(self.screen_width, self.screen_height)
        
        # Debug flag
        self.debug = hasattr(config, 'DEBUG') and config.DEBUG
        
        # Dynamic margin adjustment
        self.dynamic_margin = hasattr(config, 'DYNAMIC_MARGIN_ADJUSTMENT') and config.DYNAMIC_MARGIN_ADJUSTMENT
        self.hand_position_history = []
        self.margin_history_size = 30
        self.last_margin_update = time.time()
        self.margin_update_interval = 1.0  # seconds
        self.current_margin = config.MARGIN
        self.min_margin = 20
        self.max_margin = 100
        
        # Click control
        self.click_cooldown = click_cooldown
        self.last_click_time = 0
        self.is_dragging = False
        self.drag_start_pos = None
        
        # Dynamic speed control
        self.prev_hand_pos = None
        self.prev_time = time.time()
        self.min_speed_multiplier = 0.7  # Increased from 0.5 for better responsiveness
        self.max_speed_multiplier = 4.0  # Increased from 3.0 for faster movement
        
        # Scrolling
        self.scroll_cooldown = hasattr(config, 'SCROLL_COOLDOWN') and config.SCROLL_COOLDOWN or 0.05  # seconds between scroll actions
        self.last_scroll_time = 0
        self.scroll_speed = config.SCROLL_SPEED if hasattr(config, 'SCROLL_SPEED') else 2
        self.prev_scroll_y = None
        self.prev_scroll_x = None
        
        print(f"Mouse Controller initialized with screen dimensions: {self.screen_width}x{self.screen_height}")
        print(f"Effective screen area: {self.effective_width}x{self.effective_height}")
        print(f"Smoothing factor: {smoothing_factor}, Click cooldown: {click_cooldown}s")
        print(f"Scroll speed: {self.scroll_speed}, Scroll cooldown: {self.scroll_cooldown}s")
    
    def smooth_move(self, x, y):
        """
        Move the mouse cursor with adaptive smoothing for better responsiveness.
        
        Args:
            x (float): Normalized x-coordinate (0-1).
            y (float): Normalized y-coordinate (0-1).
            
        Returns:
            None
        """
        # Get screen dimensions
        screen_width, screen_height = self.screen_width, self.screen_height
        
        # Apply margin (avoid edges of screen)
        x = max(self.margin_factor, min(1 - self.margin_factor, x))
        y = max(self.margin_factor, min(1 - self.margin_factor, y))
        
        # Simple direct conversion to screen coordinates
        screen_x = int(x * screen_width)
        screen_y = int(y * screen_height)
        
        # Apply adaptive smoothing for better control
        if self.prev_x is not None and self.prev_y is not None:
            # Calculate distance for adaptive smoothing
            distance = ((screen_x - self.prev_x)**2 + (screen_y - self.prev_y)**2)**0.5
            
            # Dynamic smoothing based on movement speed
            # For larger movements (faster hand motion), use less smoothing
            # For smaller movements (precision), use more smoothing
            if distance > 100:  # Fast movement
                smooth_factor = 0.2  # 80% new position, 20% old position (more responsive)
            elif distance > 40:  # Medium movement
                smooth_factor = 0.3  # 70% new position, 30% old position
            else:  # Slow, precise movement
                smooth_factor = 0.4  # 60% new position, 40% old position (more stable)
            
            # Apply the adaptive smoothing
            smooth_x = int(self.prev_x * smooth_factor + screen_x * (1 - smooth_factor))
            smooth_y = int(self.prev_y * smooth_factor + screen_y * (1 - smooth_factor))
            
            # Move directly to position
            try:
                pyautogui.moveTo(smooth_x, smooth_y)
                self.prev_x, self.prev_y = smooth_x, smooth_y
                self.prev_norm_x, self.prev_norm_y = x, y
            except Exception as e:
                if self.debug:
                    print(f"Error moving cursor: {e}")
        else:
            # First movement, no smoothing needed
            try:
                pyautogui.moveTo(screen_x, screen_y)
                self.prev_x, self.prev_y = screen_x, screen_y
                self.prev_norm_x, self.prev_norm_y = x, y
            except Exception as e:
                if self.debug:
                    print(f"Error moving cursor: {e}")
    
    def _update_dynamic_margin(self):
        """
        Update margin dynamically based on the range of hand movement.
        Smaller movements = smaller margin for more precision.
        """
        if len(self.hand_position_history) < 10:
            return
        
        # Find min/max coordinates in history
        min_x = min(p[0] for p in self.hand_position_history)
        max_x = max(p[0] for p in self.hand_position_history)
        min_y = min(p[1] for p in self.hand_position_history)
        max_y = max(p[1] for p in self.hand_position_history)
        
        # Calculate hand movement range (normalized 0-1)
        x_range = max_x - min_x
        y_range = max_y - min_y
        movement_range = max(x_range, y_range)
        
        # Adjust margin based on movement range
        # Small range = small margin for precision
        # Large range = larger margin for easier reaching edges
        if movement_range < 0.4:  # Small movement
            target_margin = max(self.min_margin, int(80 * movement_range) + 20)
        else:  # Large movement
            target_margin = min(self.max_margin, int(40 * movement_range) + 40)
            
        # Smooth transition to new margin
        self.current_margin = int(0.8 * self.current_margin + 0.2 * target_margin)
        
        # Ensure margin is within valid range
        self.current_margin = max(self.min_margin, min(self.max_margin, self.current_margin))
    
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
        
        # Ensure amount is an integer and within reasonable bounds
        amount = max(1, min(10, int(amount)))
        
        # Scroll up or down
        try:
            if direction == "up":
                pyautogui.scroll(amount)  # Positive for up
            else:
                pyautogui.scroll(-amount)  # Negative for down
                
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
        
        # Ensure amount is an integer and within reasonable bounds
        amount = max(1, min(10, int(amount)))
        
        # Scroll left or right
        try:
            if direction == "left":
                pyautogui.hscroll(-amount)  # Negative for left
            else:
                pyautogui.hscroll(amount)  # Positive for right
                
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
        if self.prev_scroll_y is None or self.prev_scroll_x is None:
            self.prev_scroll_y = y_position
            self.prev_scroll_x = x_position
            return False, None
        
        # Calculate changes in position
        delta_y = y_position - self.prev_scroll_y
        delta_x = x_position - self.prev_scroll_x
        
        # Define thresholds for movement - lower threshold for better responsiveness
        threshold = 0.005  # Reduced threshold for detecting intentional movement
        
        # Set scroll speed - start with base speed
        base_scroll_amount = self.scroll_speed
        
        # Dynamic scroll speed based on movement magnitude
        # Larger movements = faster scrolling
        movement_magnitude = max(abs(delta_x), abs(delta_y))
        scroll_amount = min(10, int(base_scroll_amount * (1 + movement_magnitude * 20)))
        
        # Ensure minimum scroll amount
        scroll_amount = max(2, scroll_amount)
        
        # Track if we scrolled
        scrolled = False
        direction = None
        
        # Determine if vertical or horizontal scrolling should be performed
        # Prioritize the larger movement
        if abs(delta_y) > abs(delta_x) and abs(delta_y) > threshold:
            # Vertical scroll
            if delta_y < 0:
                # Moving hand up = scroll up
                if self.scroll_vertical("up", scroll_amount):
                    scrolled = True
                    direction = "up"
            else:
                # Moving hand down = scroll down
                if self.scroll_vertical("down", scroll_amount):
                    scrolled = True
                    direction = "down"
        elif abs(delta_x) > threshold:
            # Horizontal scroll
            if delta_x < 0:
                # Moving hand left = scroll left
                if self.scroll_horizontal("left", scroll_amount):
                    scrolled = True
                    direction = "left"
            else:
                # Moving hand right = scroll right
                if self.scroll_horizontal("right", scroll_amount):
                    scrolled = True
                    direction = "right"
        
        # Update previous positions to track relative movement
        # If we scrolled, update completely; if not, do gradual update to reduce jitter
        if scrolled:
            self.prev_scroll_x = x_position
            self.prev_scroll_y = y_position
        else:
            # Gradual update to reduce noise without losing responsiveness
            self.prev_scroll_x = 0.9 * self.prev_scroll_x + 0.1 * x_position
            self.prev_scroll_y = 0.9 * self.prev_scroll_y + 0.1 * y_position
            
        return scrolled, direction
    
    def click(self):
        """
        Perform a left mouse click without moving the cursor.
        
        Returns:
            bool: True if click was performed, False if on cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_click_time < self.click_cooldown:
            return False
        
        try:
            # Click at current position
            pyautogui.click()
            self.last_click_time = current_time
            return True
        except Exception as e:
            print(f"Click error: {e}")
            return False
    
    def right_click(self):
        """
        Perform a right mouse click without moving the cursor.
        
        Returns:
            bool: True if click was performed, False if on cooldown
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_click_time < self.click_cooldown:
            return False
        
        try:
            # Right click at current position
            pyautogui.rightClick()
            self.last_click_time = current_time
            return True
        except Exception as e:
            print(f"Right click error: {e}")
            return False
    
    def double_click(self):
        """
        Perform a double-click without moving the cursor.
        
        Returns:
            bool: True if the double-click was performed, False otherwise
        """
        # Check cooldown to prevent rapid consecutive clicks
        current_time = time.time()
        if current_time - self.last_click_time < self.click_cooldown:
            return False
        
        try:
            # Perform double-click at current position
            pyautogui.doubleClick()
            self.last_click_time = current_time
            return True
        except Exception as e:
            if self.debug:
                print(f"Double-click error: {e}")
            return False
    
    def start_drag(self, x, y):
        """
        Start a drag operation at the current cursor position.
        
        Args:
            x (int): Screen x-coordinate to start drag
            y (int): Screen y-coordinate to start drag
            
        Returns:
            bool: True if drag started, False otherwise
        """
        try:
            # First, move cursor to position (important for accurate clicking)
            pyautogui.moveTo(x, y)
            
            # Wait a small amount of time for the OS to register the position
            time.sleep(0.1)
            
            # Store current position for reference
            self.drag_start_pos = (x, y)
            self.prev_x, self.prev_y = x, y
            
            # Perform an initial click to select the item
            pyautogui.click(x, y)
            
            # Small delay after click before starting drag
            time.sleep(0.2)
            
            # Start the drag
            pyautogui.mouseDown(button='left')
            self.is_dragging = True
            
            if self.debug:
                print(f"Starting drag at: {x}, {y}")
            
            return True
        except Exception as e:
            print(f"Error starting drag: {e}")
            self.is_dragging = False
            return False
    
    def continue_drag(self, x, y):
        """
        Continue a drag operation by moving the cursor to a new position.
        
        Args:
            x (int): Screen x-coordinate to drag to
            y (int): Screen y-coordinate to drag to
            
        Returns:
            bool: True if drag continued, False otherwise
        """
        if not self.is_dragging:
            return False
        
        try:
            # Calculate movement delta for smoother motion
            dx = x - self.prev_x
            dy = y - self.prev_y
            
            # Only move if there's significant motion
            if abs(dx) > 2 or abs(dy) > 2:
                # Move to new position while holding mouse button
                pyautogui.moveTo(x, y, duration=0.01)  # Small duration for smoother movement
                self.prev_x, self.prev_y = x, y
                
                # Calculate distance dragged
                if self.drag_start_pos:
                    drag_distance = ((x - self.drag_start_pos[0])**2 + (y - self.drag_start_pos[1])**2)**0.5
                    if self.debug:
                        print(f"Dragging distance: {drag_distance:.1f} pixels")
            
            return True
        except Exception as e:
            print(f"Error continuing drag: {e}")
            return False
    
    def stop_drag(self, x=None, y=None):
        """
        Stop a drag operation and release the mouse button.
        
        Args:
            x (int, optional): Final x-coordinate to end drag, or current position if None.
            y (int, optional): Final y-coordinate to end drag, or current position if None.
            
        Returns:
            bool: True if drag ended successfully, False otherwise
        """
        if not self.is_dragging:
            return False
        
        try:
            # Move to final position if provided
            if x is not None and y is not None:
                # Move with a small duration for more accurate final positioning
                pyautogui.moveTo(x, y, duration=0.05)
                self.prev_x, self.prev_y = x, y
            
            # Small delay to ensure OS registers final position
            time.sleep(0.1)
            
            # Release mouse button
            pyautogui.mouseUp(button='left')
            
            # Additional click at the end position to ensure drop works properly
            if x is not None and y is not None:
                time.sleep(0.05)
                pyautogui.click(x, y)
            
            self.is_dragging = False
            self.drag_start_pos = None
            
            if self.debug:
                print(f"Drag completed at: {x}, {y}")
            
            return True
        except Exception as e:
            print(f"Error stopping drag: {e}")
            self.is_dragging = False
            return False
    
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
        # Use dynamic margin if available, otherwise config margin if not specified
        if margin is None:
            if self.dynamic_margin:
                margin = self.current_margin
            else:
                margin = config.MARGIN
            
        # Apply margins
        in_min_x, in_max_x = margin, input_width - margin
        in_min_y, in_max_y = margin, input_height - margin
        
        # Map to effective screen coordinates
        screen_x = max(0, min(self.effective_width - 1, 
                             self._map_range(x, in_min_x, in_max_x, 0, self.effective_width)))
        screen_y = max(0, min(self.effective_height - 1, 
                             self._map_range(y, in_min_y, in_max_y, 0, self.effective_height)))
        
        # Add offset to map to actual screen coordinates
        final_x = self.screen_offset_x + screen_x
        final_y = self.screen_offset_y + screen_y
        
        return final_x, final_y
    
    def _map_range(self, value, in_min, in_max, out_min, out_max):
        """
        Map a value from one range to another.
        
        Args:
            value (float): The value to map
            in_min, in_max (float): Input range
            out_min, out_max (float): Output range
            
        Returns:
            float: Mapped value
        """
        # Check for division by zero
        if in_max == in_min:
            return out_min
            
        # Map the value
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    def get_normalized_position(self):
        """
        Get the current mouse position in normalized coordinates (0-1).
        
        Returns:
            tuple: (x, y) normalized coordinates or (None, None) if no position yet
        """
        # Return stored normalized position if available
        if hasattr(self, 'prev_norm_x') and hasattr(self, 'prev_norm_y') and self.prev_norm_x is not None and self.prev_norm_y is not None:
            return (self.prev_norm_x, self.prev_norm_y)
        
        # If no normalized coordinates stored yet, but we have screen coordinates
        if self.prev_x is not None and self.prev_y is not None:
            try:
                # Calculate margin values
                margin_x = self.current_margin if hasattr(self, 'current_margin') else config.MARGIN
                margin_y = margin_x  # Use same margin for both axes
                
                # Convert screen coordinates back to normalized
                norm_x = (self.prev_x - margin_x) / (self.screen_width - 2 * margin_x)
                norm_y = (self.prev_y - margin_y) / (self.screen_height - 2 * margin_y)
                
                # Clamp values to 0-1 range
                norm_x = max(0, min(1, norm_x))
                norm_y = max(0, min(1, norm_y))
                
                # Store for future reference
                self.prev_norm_x = norm_x
                self.prev_norm_y = norm_y
                
                return (norm_x, norm_y)
            except Exception as e:
                print(f"Error calculating normalized position: {e}")
                return (None, None)
        
        return (None, None) 