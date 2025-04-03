#!/usr/bin/env python3
import pyautogui
import time

class MouseController:
    def __init__(self, smoothing_factor=0.5, click_cooldown=0.5):
        """
        Initialize the mouse controller.
        
        Args:
            smoothing_factor: Factor for smooth cursor movement (0-1)
            click_cooldown: Cooldown time between clicks in seconds
        """
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Disable pyautogui failsafe (moving mouse to corner won't stop execution)
        pyautogui.FAILSAFE = False
        
        # Mouse control parameters
        self.smoothing_factor = smoothing_factor
        self.click_cooldown = click_cooldown
        self.last_click_time = 0
        
        # Scrolling parameters
        self.scroll_mode = False
        self.prev_scroll_pos = None
        self.scroll_cooldown = 0
        self.SCROLL_COOLDOWN_FRAMES = 3
        
        # Movement parameters
        self.prev_cursor_pos = None
        
    def move_cursor(self, x, y, duration=0.1):
        """
        Move the cursor to the specified screen coordinates.
        
        Args:
            x: X coordinate (normalized 0-1)
            y: Y coordinate (normalized 0-1)
            duration: Time taken for movement
        """
        # Map normalized coordinates to screen dimensions
        screen_x = int(x * self.screen_width)
        screen_y = int(y * self.screen_height * 0.8)  # Scale Y to reduce vertical movement range
        
        # Move the cursor
        pyautogui.moveTo(screen_x, screen_y, duration=duration)
        
        return screen_x, screen_y
    
    def left_click(self):
        """Perform a left click if cooldown has passed"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            pyautogui.click()
            self.last_click_time = current_time
            return True
        return False
    
    def right_click(self):
        """Perform a right click if cooldown has passed"""
        current_time = time.time()
        if current_time - self.last_click_time > self.click_cooldown:
            pyautogui.rightClick()
            self.last_click_time = current_time
            return True
        return False
    
    def scroll(self, x, y):
        """
        Handle scrolling based on hand movement.
        
        Args:
            x: Current X coordinate (screen pixels)
            y: Current Y coordinate (screen pixels)
            
        Returns:
            action_performed: Whether a scroll action was performed
            direction: Direction of scroll ('horizontal', 'vertical', or None)
        """
        if not self.scroll_mode:
            self.scroll_mode = True
            self.prev_scroll_pos = (x, y)
            return False, None
        
        # Skip if we're in cooldown period
        if self.scroll_cooldown > 0:
            self.scroll_cooldown -= 1
            return False, None
        
        # Handle scrolling based on hand movement
        if self.prev_scroll_pos:
            dx = x - self.prev_scroll_pos[0]
            dy = y - self.prev_scroll_pos[1]
            
            action_performed = False
            direction = None
            
            # Threshold to avoid small movements
            if abs(dx) > 15:
                # Horizontal scrolling
                pyautogui.hscroll(-int(dx/10))  # Negative to match natural scrolling direction
                action_performed = True
                direction = "horizontal"
                self.scroll_cooldown = self.SCROLL_COOLDOWN_FRAMES
            
            if abs(dy) > 15:
                # Vertical scrolling
                pyautogui.vscroll(-int(dy/10))  # Negative to match natural scrolling direction
                action_performed = True
                direction = "vertical"
                self.scroll_cooldown = self.SCROLL_COOLDOWN_FRAMES
            
            self.prev_scroll_pos = (x, y)
            return action_performed, direction
        
        return False, None
    
    def exit_scroll_mode(self):
        """Exit scroll mode"""
        self.scroll_mode = False
        self.prev_scroll_pos = None 