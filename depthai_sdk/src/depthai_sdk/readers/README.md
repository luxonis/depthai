## DepthAI readers

Here we have helper classes that read recorded files. For each recorded file (eg. mp4, mjpeg, bag, h265, mcap, etc.) we create a new object.

- `videocap_reader.py` uses `cv2.VideoCapture()` class which reads mp4, mjpeg, lossless mjpeg, and h265.
- `rosbag_reader.py` reads from rosbags (.bag) which is mainly used to record depth files.
- `mcap_reader.py` reads from [Foxglove](https://foxglove.dev/)'s [mcap container](https://github.com/foxglove/mcap).
- `image_reader.py` uses `cv2.imread()` class to read all popular image files (png, jpg, bmp, webp, etc.).