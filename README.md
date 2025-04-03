# Virtual Mouse

A Python-based virtual mouse that tracks hand movements using computer vision to control cursor movement, clicks, and scrolling with high precision and smooth performance.

## Features

- **Hand Tracking**: Accurately tracks hand landmarks in various lighting conditions using MediaPipe
- **Cursor Control**: Smooth and responsive cursor movement based on hand position
- **Click Actions**:
  - Left Click: Join index finger and thumb
  - Right Click: Form a peace sign (index and middle finger extended)
- **Scrolling**:
  - Vertical Scroll: Move hand up/down with three fingers extended
  - Horizontal Scroll: Move hand left/right with three fingers extended
- **Drag and Drop**: Pinch and hold (index finger and thumb) to activate drag mode
- **Customizable**: All parameters can be adjusted in the config file or through command-line arguments

## Requirements

- Python 3.7+
- Webcam

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/virtual_mouse.git
   cd virtual_mouse
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the virtual mouse:
```
python main.py
```

### Command Line Options

You can customize the behavior using command-line arguments:

```
python main.py --camera 0 --smoothing 7 --no-debug
```

Available options:
- `--camera`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--detection-confidence`: Hand detection confidence threshold (default: 0.7)
- `--tracking-confidence`: Hand tracking confidence threshold (default: 0.7)
- `--max-hands`: Maximum number of hands to detect (default: 1)
- `--smoothing`: Mouse movement smoothing factor (default: 7)
- `--margin`: Margin from frame edges (default: 100)
- `--no-debug`: Disable debug information display
- `--no-landmarks`: Disable hand landmark display

## Controls

### Gestures

- **Move Cursor**: Move your index finger
- **Left Click**: Join index finger and thumb
- **Right Click**: Form a peace sign (index and middle finger extended)
- **Scroll**: Extend three fingers and move hand up/down or left/right
- **Drag and Drop**: Pinch (join index finger and thumb) and hold for a moment

### Keyboard Shortcuts

While the application is running, you can use these keyboard shortcuts:
- `q`: Quit the application
- `d`: Toggle debug information display
- `h`: Toggle hand landmark display

## Configuration

You can customize the behavior by editing `config.py`. The file contains settings for:

- Camera properties
- Hand tracking parameters
- Mouse movement sensitivity
- Click and gesture thresholds
- UI appearance

## How It Works

The application uses the following components:

1. **HandTracker**: Uses MediaPipe to detect hand landmarks and implements gesture recognition
2. **MouseController**: Handles mouse actions like movement, clicking, and scrolling
3. **VirtualMouse**: Connects hand tracking with mouse control and manages the application flow

## Troubleshooting

If you experience poor tracking:
- Ensure good lighting conditions
- Position your hand within the camera frame
- Adjust detection confidence in config.py
- Try changing the camera resolution

If mouse movements are too sensitive or jumpy:
- Increase the smoothing factor in config.py
- Increase the margin value to reduce the active area

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control 