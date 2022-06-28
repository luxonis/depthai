import cv2
import sys

if len(sys.argv) <= 1:
    raise Exception("Specify the path to the video file (.mp4, .mjpeg, .h265, etc.) like `videoCap.py color.mjpeg`")

cap = cv2.VideoCapture(sys.argv[1])
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord("q"): break