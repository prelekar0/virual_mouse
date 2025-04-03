# Virtual Mouse

A Python-based virtual mouse that tracks hand movements using computer vision to control cursor movement, clicks, and scrolling with high precision and smooth performance.

## Features

- **Hand Tracking**: Accurately tracks hand landmarks in various lighting conditions using MediaPipe
- **Cursor Control**: Smooth and responsive cursor movement based on hand position with dynamic speed adjustment
- **Click Actions**:
  - Left Click: Middle finger extended (index finger down)
  - Right Click: Index finger extended (middle finger down)
  - Double Click: Index and middle fingers close together
- **Scrolling**:
  - Vertical Scroll: Move hand up/down in scroll mode (thumb, index, middle fingers up)
  - Horizontal Scroll: Move hand left/right in scroll mode (thumb, index, middle fingers up)
- **Drag and Drop**: Pinch and hold gesture to activate drag mode
- **Customizable**: All parameters can be adjusted in the config file or through command-line arguments
- **Debug Mode**: Real-time visual feedback and customizable parameters

## Requirements

- Python 3.7+
- Webcam
- Operating System: Windows, macOS, or Linux

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/prelekar0/virual_mouse.git
   cd virual_mouse
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

4. Verify your setup:
   ```
   python virtual_mouse.py --show-landmarks --debug
   ```

## Usage

Run the virtual mouse:
```
python virtual_mouse.py
```

### Command Line Options

You can customize the behavior using command-line arguments:

```
python virtual_mouse.py --camera 0 --smoothing 7 --debug --show-landmarks
```

Available options:
- `--camera`: Camera index (default: 0)
- `--width`: Camera width (default: 640)
- `--height`: Camera height (default: 480)
- `--detection-confidence`: Hand detection confidence threshold (default: 0.75)
- `--tracking-confidence`: Hand tracking confidence threshold (default: 0.75)
- `--max-hands`: Maximum number of hands to detect (default: 1)
- `--smoothing`: Mouse movement smoothing factor (default: 5)
- `--debug`: Enable debug information display
- `--show-landmarks`: Enable hand landmark display

## Controls

### Gestures

| Gesture | Description | Action |
|---------|-------------|--------|
| Open Hand | All fingers extended | Neutral position |
| Peace Sign | Index and middle fingers extended | Move cursor |
| Middle Finger Only | Middle finger extended, others down | Left click |
| Index Finger Only | Index finger extended, others down | Right click |
| Index & Middle Close | Index and middle fingers extended and close together | Double click |
| Three Fingers Up | Thumb, index, and middle fingers extended | Scroll mode |

### Keyboard Shortcuts

While the application is running, you can use these keyboard shortcuts:
- `ESC`: Quit the application
- `D`: Toggle debug information display
- `H`: Toggle hand landmark display

## Configuration

You can customize the behavior by editing `config.py`. The file contains settings for:

- Camera properties
- Hand tracking parameters
- Mouse movement sensitivity and smoothing
- Click and gesture thresholds
- UI appearance and debug settings

## How It Works

The application uses the following components:

1. **HandTracker** (`hand_tracking.py`): Uses MediaPipe to detect hand landmarks and implements gesture recognition algorithms
2. **MouseController** (`mouse_controller.py`): Handles mouse actions including smoothed movement, clicking, scrolling, and dragging
3. **VirtualMouse** (`virtual_mouse.py`): Connects hand tracking with mouse control and manages the application flow
4. **Configuration** (`config.py`): Centralized configuration for all application parameters

## Troubleshooting

### Hand Detection Issues
- Ensure good, consistent lighting conditions (avoid backlighting)
- Position your hand within the camera frame at about 30-60 cm from the camera
- Use a plain background for better detection
- Adjust the detection confidence threshold (try values between 0.6-0.8)
- Ensure your hand occupies about 1/3 of the frame for optimal detection

### Mouse Movement Problems
- If movements are too sensitive: Increase the smoothing factor (7-12)
- If movements are too slow: Reduce the smoothing factor (3-5)
- Adjust the margin value to change the active area size
- Try disabling dynamic speed adjustment in `mouse_controller.py` if movements are inconsistent

### System Performance
- If the application is running slowly, try:
  - Reducing the camera resolution (e.g., 480x360)
  - Processing fewer frames (modify `PROCESS_EVERY_N_FRAMES` in config.py)
  - Closing other resource-intensive applications

### Calibration
For better results, you may need to adjust parameters based on your hand size and lighting conditions:
1. Run with the `--debug` flag
2. Use the trackbars in the Controls window to adjust parameters in real-time
3. Once you find good values, update them in `config.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyAutoGUI](https://pyautogui.readthedocs.io/) for mouse control
- [Numpy](https://numpy.org/) for numerical operations 