# Virtual Mouse Configuration

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Hand tracking settings
DETECTION_CONFIDENCE = 0.75  # Increased for more reliable detection
TRACKING_CONFIDENCE = 0.75   # Increased for more reliable tracking
MAX_HANDS = 1
STATIC_MODE = False
LANDMARK_SMOOTHING = True
LANDMARK_HISTORY_LENGTH = 3  # Reduced for even faster response

# Mouse movement settings
SMOOTHING_FACTOR = 5  # Reduced for faster cursor response
MARGIN = 80  # Reduced margin to use more of the camera frame

# Click settings
CLICK_COOLDOWN = 0.3  # seconds
DOUBLE_CLICK_INTERVAL = 0.5  # seconds
PINCH_DISTANCE_THRESHOLD = 50  # Increased for easier clicking

# Scroll settings
SCROLL_THRESHOLD = 8  # Reduced for easier scrolling activation
SCROLL_SPEED = 4  # Increased for more responsive scrolling

# Drag settings
DRAG_THRESHOLD = 12  # Reduced for easier drag activation

# Performance settings
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for better responsiveness
BOOST_CONTRAST = True  # Enhance image contrast for better detection
CONTRAST_ALPHA = 1.3  # Increased contrast
CONTRAST_BETA = 15  # Increased brightness

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
        "pinch_distance": 50  # Maximum distance between thumb and index for click
    },
    "right_click": {
        # Configuration for peace sign (index and middle up, others down)
    },
    "scroll": {
        # Configuration for three fingers up
    },
    "drag": {
        "hold_frames": 12  # Frames to hold before activating drag
    }
}

# Debug settings
DEBUG = True
SHOW_LANDMARKS = True
SHOW_FPS = True
SHOW_HELP = True 