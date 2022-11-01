from pathlib import Path
import depthai as dai
import cv2


class VideoWriter:
    file = None
    _fps: float
    _path: str

    def __init__(self, folder: Path, name: str, fourcc: str, fps: float):
        self._path = str(folder / f"{name}.avi")
        self._fps = fps

    def _create_file(self, frame: dai.ImgFrame):
        w = frame.getWidth()
        h = frame.getHeight()

        if frame.getType() == dai.ImgFrame.Type.RAW16:  # Depth
            fourcc = "Y16 "
        elif frame.getType() == dai.ImgFrame.Type.RAW8:  # Mono Cams
            fourcc = "GREY"
        else:
            fourcc = "I420"

        self.file = cv2.VideoWriter(self._path,
                                    cv2.VideoWriter_fourcc(*fourcc),
                                    self._fps,
                                    (w, h),
                                    isColor=fourcc == "I420")

    def close(self):
        self.file.release()

    def write(self, frame: dai.ImgFrame):
        if self.file is None:
            self._create_file(frame)

        self.file.write(frame.getCvFrame())
