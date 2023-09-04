import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import depthai as dai

try:
    import cv2
except ImportError:
    cv2 = None

from depthai_sdk.readers.abstract_reader import AbstractReader
from depthai_sdk.components.parser import parse_camera_socket

_videoExt = ['.mjpeg', '.avi', '.mp4', '.h265', '.h264']


class VideoCapReader(AbstractReader):
    """
    Reads stream from mp4, mjpeg, h264, h265
    """

    def __init__(self, path: Path, loop: bool = False) -> None:
        self.videos: Dict[str, Any] = {}
        self._closed = False

        # self.initialFrames: Dict[str, Any] = dict()
        # self.shapes: Dict[str, Tuple[int, int]] = dict()
        # self.readers: Dict[str, cv2.VideoCapture] = dict()
        self._is_looped = loop

        if path.is_file():
            stream_name = path.stem if (path.stem in ['left', 'right']) else 'color'
            self.videos[stream_name] = {
                'reader': cv2.VideoCapture(str(path)),
                'socket': dai.CameraBoardSocket.CAM_A
            }
        else:
            for fileName in os.listdir(str(path)):
                f_name, ext = os.path.splitext(fileName)
                if ext not in _videoExt:
                    continue

                # Check if name of the file starts with left.. right.., or CameraBoardSocket
                if f_name.startswith('CAM_'):
                    # Remove everything after CAM_x
                    f_name = f_name[:5]

                socket = None
                try:
                    socket = parse_camera_socket(f_name)
                except ValueError:
                    # Invalid file name
                    pass

                # TODO: avoid changing stream names, just use socket
                # stream = str(socket)
                # if socket == dai.CameraBoardSocket.CAM_A:
                #     stream = 'color'
                # elif socket == dai.CameraBoardSocket.CAM_B:
                #     stream = 'left'
                # elif socket == dai.CameraBoardSocket.CAM_C:
                #     stream = 'right'
                self.videos[f_name.lower()] = {
                    'reader': cv2.VideoCapture(str(path / fileName)),
                    'socket': socket
                }

        for name, video in self.videos.items():
            ok, f = video['reader'].read()
            video['shape'] = (
                f.shape[1],
                f.shape[0]
            )
            video['is_color'] = len(f.shape) == 3
            video['initialFrame'] = f

    def read(self):
        if self._closed:
            return False
        frames = dict()
        for name, video in self.videos.items():
            if video['initialFrame'] is not None:
                frames[name] = video['initialFrame'].copy()
                video['initialFrame'] = None

            if not video['reader'].isOpened():
                return False

            ok, frame = video['reader'].read()
            if not ok and self._is_looped:
                video['reader'].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = video['reader'].read()
            elif not ok:
                return False

            frames[name] = frame

        return frames

    def set_loop(self, loop: bool):
        self._is_looped = loop

    def getStreams(self) -> List[str]:
        return [name for name in self.videos]

    def getShape(self, name: str) -> Tuple[int, int]:
        shape = self.videos[name.lower()]['shape']
        return shape

    def get_socket(self, name: str) -> Optional[dai.CameraBoardSocket]:
        return self.videos[name.lower()]['socket']

    def close(self):
        [r['reader'].release() for _, r in self.videos.items()]
        self._closed = True

    def disableStream(self, name: str):
        if name.lower() in self.videos:
            self.videos.pop(name.lower())

    def get_message_size(self, name: str) -> int:
        video = self.videos[name.lower()]
        return video['shape'][0] * video['shape'][1] * (3 if video['is_color'] else 1)
