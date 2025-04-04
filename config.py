# Virtual Mouse Configuration

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60

# Hand tracking settings
DETECTION_CONFIDENCE = 0.6
TRACKING_CONFIDENCE = 0.7
MAX_HANDS = 1
STATIC_MODE = False
LANDMARK_SMOOTHING = True
LANDMARK_HISTORY_LENGTH = 2  # Reduced for faster response

# Mouse movement settings
SMOOTHING_FACTOR = 0.2
MARGIN = 10  # Pixels from screen edge

# Click settings
CLICK_COOLDOWN = 0.4
DOUBLE_CLICK_INTERVAL = 0.4
PINCH_DISTANCE_THRESHOLD = 40  # Reduced for more precise detection
CLICK_STABILIZATION_FRAMES = 5  # Increased frames to stabilize after clicking
CLICK_POSITION_TOLERANCE = 5  # Pixel tolerance for movement during clicking

# Scroll settings
SCROLL_THRESHOLD = 0.008
SCROLL_SPEED = 3
SCROLL_COOLDOWN = 0.05

# Drag settings
DRAG_THRESHOLD = 10  # Reduced for easier drag activation

# Performance settings
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for smooth movement
MAX_FPS = 60  # Higher FPS for smoother tracking
ENABLE_OPTIMIZATION = True

# Contrast and image processing
BOOST_CONTRAST = False  # Disable contrast enhancement for better performance
CONTRAST_ALPHA = 1.0  # No contrast adjustment
CONTRAST_BETA = 0  # No brightness adjustment

# UI settings
COLORS = {
    "cursor": (0, 255, 0),       # Green
    "left_click": (0, 255, 0),   # Green
    "right_click": (0, 0, 255),  # Red
    "scroll": (255, 0, 0),       # Blue
    "drag": (255, 165, 0),       # Orange
    "text": (0, 0, 0)            # Black
}

# Gesture thresholds
GESTURES = {
    "left_click": {
        "pinch_distance": 60  # Increased distance for easier clicking
    },
    "right_click": {
        # Configuration for peace sign (index and middle up, others down)
    },
    "scroll": {
        # Configuration for three fingers up
    },
    "drag": {
        "hold_frames": 10  # Reduced frames to hold before activating drag
    }
}

# Screen adjustment for better coverage
SCREEN_COVERAGE_FACTOR = 1.0  # Use full screen
DYNAMIC_MARGIN_ADJUSTMENT = True  # Dynamically adjust margins based on hand position

# Debug settings
DEBUG = False
SHOW_LANDMARKS = True
SHOW_FPS = True
SHOW_HELP = True 