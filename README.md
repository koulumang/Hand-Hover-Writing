## Hand-Hover-Writing

Write in the air using hand gestures tracked by your webcam. No touching required.

## Features

- **Automatic hand detection** using MediaPipe
- **Pinch gesture control** - pinch thumb and index finger to draw, release to stop
- **Smooth line drawing** with connected strokes
- **Performance optimizations** - FPS counter, line smoothing, optimized resolution
- **Keyboard controls:**
  - `c` - clear canvas
  - `s` - save drawing as PNG
  - `ESC` - quit

## Algorithms Used

### Hand Tracking
- **MediaPipe Hands** - Google's ML solution for real-time hand landmark detection
  - Detects 21 3D hand landmarks per hand
  - Uses palm detection model + hand landmark model
  - Runs on-device with TensorFlow Lite

### Gesture Recognition
- **Euclidean distance** between thumb tip (landmark 4) and index finger tip (landmark 8)
- Pinch detected when distance < 40 pixels
- Binary state: drawing (pinched) vs not drawing (released)

### Line Smoothing
- **Moving average filter** with buffer size of 5 points
- Reduces hand tremor and jitter
- Formula: `smoothed_point = mean(last_5_points)`
- Buffer cleared on pen-up to avoid stroke lag

### Performance Optimization
- **640x480 resolution** for faster processing
- **Rolling FPS calculation** averaged over 30 frames
- **Deque with maxlen=1000** for memory-efficient point storage
- **Mirrored display** for natural writing experience

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-contrib-python mediapipe numpy
```

Download the hand tracking model:
```bash
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

Run the improved version:
```bash
python Writing_improved.py
```

Or the original tracker version:
```bash
python Writing.py
```

## How to Use

1. Show your hand to the camera
2. Pinch thumb and index finger together to start drawing (green circle)
3. Move your hand while pinched to write
4. Release pinch to stop drawing (red circle)
5. Press `c` to clear, `s` to save, `ESC` to quit

## Technical Details

**Original Version (Writing.py)**
- Uses OpenCV CSRT tracker
- Manual ROI selection required
- Tracks bounding box center
- No gesture control

**Improved Version (Writing_improved.py)**
- MediaPipe hand landmark detection
- Automatic hand tracking
- Gesture-based pen control
- Smoothed line rendering
- Real-time FPS display
