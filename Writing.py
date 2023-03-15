import cv2
import sys

tracker_types = ['MIL', 'KCF', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture(0)

#ignore first 20 frames to avoid blackness
for i in range(20):
    ok, frame = video.read()
ok, frame = video.read()
if not ok:
    print
    'Cannot read video file'
    sys.exit()

bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

points=[]
while True:
    ok, frame = video.read()
    if not ok:
        break
    ok, bbox = tracker.update(frame)
    if ok:
        # p1 = (int(bbox[0]), int(bbox[1]))
        # p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        x_center=int(bbox[0]) + (int(bbox[2]))/2
        y_center=int(bbox[1]) + (int(bbox[3]))/2
        print('x_center: ',x_center,' y_center: ',y_center)
        points.append([int(x_center),int(y_center)])
    else:
        cv2.putText(frame, "!!!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    for i in range(len(points)):
        cv2.circle(frame, (points[i][0], points[i][1]), 3, (0, 0, 255), 2)


    cv2.imshow("Tracking", cv2.flip(frame, 1))

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break

