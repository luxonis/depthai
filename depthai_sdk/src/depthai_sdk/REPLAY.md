# Replay module

Replay will read `depthai-recordings` and send the stream back to the OAK camera for scene reconstruction,
with optional complete stereo depth reconstruction from stereo camera streams (`left` and `right`).

```python
from depthai_sdk import Replay
depthai_recording:str = 'path/to/recording' 
replay = Replay(depthai_recording)
```

Where `depthai_recording` can be the following:
- Path to `depthai_recording` folder, recorded with [Record App](https://github.com/luxonis/depthai#record-app)
(containing `calib.json` and streams in mp4/mjpeg/h265 formats)
- Link to YouTube video, which gets downloaded
- Path to `.mcap` file
- Path to `.bag` file
- Path to ROS2 recording folder (`.db3`, `.yaml`, `calib.json`)
- Name of a [`depthai-recording` supported by SDK]() 


See Replay usage examples at [Replay record demo](https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay).

### depthai-recordings supported by SDK

We host several depthai-recordings on our servers that you can easily use in your
application. Recording will get downloaded & cached on the computer for future use.

Name | Files | Size | Note
---|---|---|---
`cars-california-01` | `color.mp4`| 21.1 MB | [Source video](https://www.youtube.com/watch?v=whXnYIgT4P0), useful for car detection / license plate recognition
`cars-california-02` | `color.mp4`| 27.5 MB | [Source video](https://www.youtube.com/watch?v=whXnYIgT4P0), useful for car detection / license plate recognition
`cars-california-03` | `color.mp4`| 19 MB   | [Source video](https://www.youtube.com/watch?v=whXnYIgT4P0), useful for license plate recognition and bicylist detection
`depth-people-counting-01` | `left.mp4`, `right.mp4`, `calib.json` | 5.8 MB  | Used by [depth-people-counting](https://github.com/luxonis/depthai-experiments/tree/master/gen2-depth-people-counting) demo