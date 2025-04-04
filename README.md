# Virtual Mouse

A computer vision-based virtual mouse controller that allows you to control your cursor and perform mouse actions using hand gestures. This application uses MediaPipe for hand tracking and translates hand movements and gestures into precise mouse control.

## Features

- **Intuitive Gesture Control**: Natural hand gestures for all mouse operations
- **Cursor Movement**: Control the cursor with index and middle fingers extended (peace sign)
- **Click Actions**:
  - Left Click: Lower only the index finger from a neutral position or from peace sign
  - Right Click: Lower only the middle finger from a neutral position or from peace sign
  - Double Click: Bring index and middle fingers close together
- **Drag & Drop**: Pinch your thumb and index finger together while keeping other fingers extended
- **Scrolling**:
  - Vertical Scroll: Move hand up/down in scroll mode (thumb, index, middle fingers up)
  - Horizontal Scroll: Move hand left/right in scroll mode (thumb, index, middle fingers up)
- **Neutral Mode**: Open hand with all fingers extended for no action
- **Smooth Performance**: Optimized tracking with adaptive smoothing for more precise control
- **Visual Feedback**: On-screen indicators for gestures and actions
- **Customizable**: Adjust sensitivity, smoothing, and other parameters via config file or command-line

## Requirements

- Python 3.7+
- Webcam
- Operating System: Windows, macOS, or Linux

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/prelekar0/virual_mouse.git
   
   ```

2. Create a virtual environment (recommended):
   ```
   # On Windows
   python -m venv myenv
   myenv\Scripts\activate

   # On macOS/Linux
   python -m venv myenv
   source myenv/bin/activate
   ```

3. Install dependencies:
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
python main.py --camera 0 --debug --show-landmarks --display-scale 1.0
```

Available options:
- `--camera`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--smoothing`: Mouse movement smoothing factor (default: 0.5)
- `--cooldown`: Time between clicks in seconds (default: 0.5)
- `--show-landmarks`: Show hand landmarks
- `--debug`: Enable debug information display
- `--display-scale`: Scale factor for display window (0.1-1.0, default: 0.7)

## Gesture Guide

| Gesture | Description | Action |
|---------|-------------|--------|
| ✋ Open Hand | All fingers extended | Neutral position (no action) |
| ✌️ Peace Sign | Index and middle fingers extended | Move cursor |
| 👇 Lower Index | From neutral, lower just the index finger | Left click |
| 👉 Lower Middle | From neutral, lower just the middle finger | Right click |
| 🤞 Fingers Close | Index and middle fingers close together | Double click |
| 👌 Pinch | Thumb and index pinch with other fingers up | Drag and drop |
| 👆 Three Fingers | Thumb, index, and middle fingers up | Scroll mode |

### Tips for Best Performance

- **Lighting**: Ensure consistent, good lighting on your hand
- **Background**: Use a plain background for better detection
- **Hand Position**: Keep your hand 30-60cm from the camera
- **Camera Angle**: Position your camera to capture your hand comfortably
- **Hand Orientation**: Keep your palm facing the camera for best recognition

### Keyboard Shortcuts

- `ESC`: Quit the application
- `D`: Toggle debug information display
- `H`: Toggle hand landmark display

## How It Works

The application consists of three main components:

1. **HandTracker** (`hand_tracking.py`): Uses MediaPipe to detect hand landmarks and implements gesture recognition
2. **MouseController** (`mouse_controller.py`): Handles mouse actions with adaptive sensitivity and smoothing
3. **VirtualMouse** (`virtual_mouse.py`): Connects hand tracking with mouse control and manages application flow

## Troubleshooting

### Hand Detection Issues
- Ensure good lighting (avoid backlighting)
- Position your hand within the camera frame
- Use a plain background
- Adjust hand position and angle to improve tracking

### Performance Issues
- Try reducing the display scale with `--display-scale 0.5`
- If movements are too sensitive, increase the smoothing factor
- If clicks are too frequent, increase the cooldown value
- If the application is lagging, close other resource-intensive applications

## Configuration

You can customize the behavior by editing `config.py`, which contains settings for:
- Camera properties
- Hand tracking parameters
- Mouse movement sensitivity
- Click and gesture thresholds
- Performance optimization settings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control
