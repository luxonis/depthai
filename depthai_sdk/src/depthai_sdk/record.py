#!/usr/bin/env python3
import array
from pathlib import Path
from queue import Queue
from threading import Thread
import depthai as dai
from enum import IntEnum

def _run(recorders, frameQ):
    """
    Start recording infinite loop
    """
    while True:
        try:
            frames = frameQ.get()
            if frames is None: # Terminate app
                break
            for name in frames:
                # Save all synced frames into files
                recorders[name].write(name, frames[name])
        except KeyboardInterrupt:
            break
    # Close all recorders - Can't use ExitStack with VideoWriter
    for n in recorders:
        recorders[n].close()
    print('Exiting store frame thread')

class Recorder(IntEnum):
    RAW = 1 # Save raw bitstream
    MP4 = 2 # Containerize into mp4 file, requires `av` library
    MCAP = 3 # To .mcap
    BAG = 4 # To ROS .bag

class Record():
    """
    This class records depthai streams from OAK cameras into different formats.
    Available formats: .h265, .mjpeg, .mp4, .mcap, .bag
    It will also save calibration .json, so depth reconstruction will 
    """

    _timelapse: int = -1

    def __init__(self, path: Path, device: dai.Device):
        """
        Args:
            path (Path): Path to the recording folder
            device (dai.Device): OAK device object
        """
        self.device = device
        self.stereo = 1 < len(device.getConnectedCameras())
        self.mxid = device.getMxId()
        self.path = self._createFolder(path, self.mxid)

        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))


    def _getRecorders(self) -> dict:
        """
        Create recorders
        """
        recorders = dict()
        save = self.save.copy()

        if self._mcap:
            if self.quality == EncodingQuality.LOW or self.quality == EncodingQuality.BEST:
                raise Exception("MCAP only supports MEDIUM and HIGH quality!") # Foxglove Studio doesn't support Lossless MJPEG
            from .recorders.mcap_recorder import McapRecorder
            rec = McapRecorder(self.path, self.device)
            rec.setPointcloud(self._pointcloud)
            for name in save:
                recorders[name] = rec
            return recorders

        if 'depth' in save:
            from .recorders.rosbag_recorder import RosbagRecorder
            recorders['depth'] = RosbagRecorder(self.path, self.device, self.getSizes())
            save.remove('depth')

        if len(save) == 0: return recorders

        else:
            try:
                # Try importing av
                from .recorders.pyav_mp4_recorder import PyAvRecorder
                rec = PyAvRecorder(self.path, self.quality, self.args.rgbFps, self.args.monoFps)
            except:
                print("'av' library is not installed, depthai-record will save raw encoded streams.")
                from .recorders.raw_recorder import RawRecorder
                rec = RawRecorder(self.path, self.quality)
        # All other streams ("color", "left", "right", "disparity") will use
        # the same Raw/PyAv recorder
        for name in save:
            recorders[name] = rec
        return recorders

    def start(self):
        """
        Start recording process. This will create and start the pipeline,
        start recording threads, and initialize all queues.
        """
        if not self.stereo: # If device doesn't have stereo camera pair
            if "left" in self.save: self.save.remove("left")
            if "right" in self.save: self.save.remove("right")
            if "disparity" in self.save: self.save.remove("disparity")
            if "depth" in self.save: self.save.remove("depth")

        if self._preview: self.save.append('preview')

        if 0 < self._timelapse:
            self.args.monoFps = 5.0
            self.args.rgbFps = 5.0


    def setTimelapse(self, timelapseSec: int):
        """
        Sets number of seconds between each frame for the timelapse mode.
        """
        self._timelapse = timelapseSec

    def _createFolder(self, path: Path, mxid: str) -> Path:
        """
        Creates recording folder
        """
        i = 0
        while True:
            i += 1
            recordings_path = path / f"{i}-{str(mxid)}"
            if not recordings_path.is_dir():
                recordings_path.mkdir(parents=True, exist_ok=False)
                return recordings_path
