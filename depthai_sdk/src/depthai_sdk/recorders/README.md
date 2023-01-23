# Recorder classes

Here we have multiple recorder classes:

- `raw_recorder.py` saves raw bitstream directly into a file (`.h265` or `.mjpeg`). Video players like VLC can't play such videos, as they aren't in a container. `cv2.VideoCapture` can read frames directly from such files.
- `pyav_mp4_recorder.py` saves bitstream directly into an .mp4 container. Simple demo of this functionality can be [found here](../../../gen2-container-encoding/). It requires [PyAV](https://github.com/PyAV-Org/PyAV) installed. All recorded streams (color,left,right,disparity) will be saved in .mp4 file and users can watch them with a video player. `BEST` quality is **lossless MJPEG**, which **isn't supported by video players**, but you can still use `cv2.VideoCapture` to watch the video.
- `rosbag_recorder.py` saves raw depth frames into rosbag (.bag) which can be imported to the [RealSense Viewer](https://www.intelrealsense.com/sdk-2/#sdk2-tools). You will need `rosbag` python library to use this recorder.

### Watch uncontainerized videos

To watch uncontainerized videos (`.h265`, `.mjpeg`) or lossless MJPEG encoded `.mp4` video, you can use `cv2.VideoCapture()`. A simple usage:

```python
import sys

import cv2

if len(sys.argv) <= 1:
    raise Exception("Specify the path to the video file (.mp4, .mjpeg, .h265, etc.) like `video_cap.py color.mjpeg`")

cap = cv2.VideoCapture(sys.argv[1])
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord("q"): break
```