import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
detector = HandLandmarker.create_from_options(options)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

points = deque(maxlen=1000)
is_drawing = False
smooth_buffer = deque(maxlen=5)
fps_buffer = deque(maxlen=30)

def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def smooth_point(point):
    smooth_buffer.append(point)
    if len(smooth_buffer) < 3:
        return point
    x = int(np.mean([p[0] for p in smooth_buffer]))
    y = int(np.mean([p[1] for p in smooth_buffer]))
    return (x, y)

while True:
    start_time = time.time()
    
    ok, frame = video.read()
    if not ok:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect_for_video(mp_image, int(cv2.getTickCount()))
    
    if results.hand_landmarks:
        hand = results.hand_landmarks[0]
        h, w, _ = frame.shape
        
        thumb_tip = hand[4]
        index_tip = hand[8]
        
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        
        distance = get_distance(thumb_pos, index_pos)
        
        if distance < 40:
            is_drawing = True
            smoothed_pos = smooth_point(index_pos)
            cv2.circle(frame, smoothed_pos, 8, (0, 255, 0), -1)
        else:
            is_drawing = False
            cv2.circle(frame, index_pos, 8, (0, 0, 255), 2)
            smooth_buffer.clear()
        
        if is_drawing:
            points.append(smooth_point(index_pos))
        else:
            points.append(None)
    
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        cv2.line(frame, points[i-1], points[i], (255, 0, 0), 3)
    
    fps = 1.0 / (time.time() - start_time)
    fps_buffer.append(fps)
    avg_fps = np.mean(fps_buffer)
    
    cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Pinch to draw | C: clear | S: save | ESC: quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Hand Writing", frame)
    
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif k == ord('c'):
        points.clear()
    elif k == ord('s'):
        cv2.imwrite('drawing.png', frame)
        print("Saved as drawing.png")

video.release()
cv2.destroyAllWindows()
