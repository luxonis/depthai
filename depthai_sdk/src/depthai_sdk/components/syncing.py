import time

import depthai as dai
from typing import Dict, List, Any, Optional, Tuple, Callable
from queue import Empty, Queue
from abc import ABC, abstractmethod
from .component import Component


class BaseSync(ABC):
    callbacks: List[Callable]  # List of callback to be called when we have new synced msgs
    queue: Queue  # Queue to which (synced/non-synced) messages will be added
    streams: List[str]  # Streams to listen for
    components: List[Component]

    def __init__(self, callbacks: List[Callable], components: List[Component]) -> None:
        self.callbacks = callbacks
        self.queue = Queue(maxsize=30)
        self.components = components

    def setup(self):
        """
        Set up the Syncing logic after the OAK is connected and all components are finalized
        """
        self.streams = []
        for comp in self.components:
            self.streams.extend([name for name, _ in comp.xouts.items()])

    @abstractmethod
    def newMsg(self, name: str, msg) -> None:
        raise NotImplementedError()

    # This approach is used as some functions (eg. imshow()) need to be called from
    # main thread, and calling them from callback thread wouldn't work.
    def checkQueue(self, block=False) -> None:
        """
        Checks queue for any available messages. If available, call callback. Non-blocking by default.
        """
        try:
            msgs = self.queue.get(block=block)
            if msgs is not None:
                for cb in self.callbacks:  # Call all callbacks
                    cb(msgs)
        except Empty:  # Queue empty
            pass


class NoSync(BaseSync):
    """
    Will call callback whenever it gets a new message
    """
    msgs: Dict[str, List[dai.Buffer]] = dict()  # List of messages

    def newMsg(self, name: str, msg) -> None:
        # Ignore frames that we aren't listening for
        if name not in self.streams: return

        if name not in self.msgs: self.msgs[name] = []

        self.msgs[name].append(msg)
        msgsAvailableCnt = [0 < len(arr) for _, arr in self.msgs.items()].count(True)

        if len(self.streams) == msgsAvailableCnt:
            # We have at least 1 msg for each stream. Get the latest, remove all others.
            ret = {}
            for name, arr in self.msgs.items():
                # print(f'len(msgs[{name}])', len(self.msgs[name]))
                self.msgs[name] = self.msgs[name][-1:]  # Remove older msgs
                # print(f'After removing - len(msgs[{name}])', len(self.msgs[name]))
                ret[name] = arr[-1]

            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            # print(time.time(),' Putting msg batch into queue. queue size', self.queue.qsize(), 'self.msgs len')

            self.queue.put(ret, block=False)

class SequenceSync(BaseSync):
    """
    This class will sync all messages based on their sequence number
    """
    msgs: Dict[str, Dict[str, dai.Buffer]] = dict()  # List of messages.
    """
    msgs = {seq: {stream_name: frame}}
    Example:
    
    msgs = {
        '1': {
            'rgb': dai.Frame(),
            'dets': dai.ImgDetections(),
        }
        '2': {
            'rgb': dai.Frame(),
            'dets': dai.ImgDetections(),
        }
    }
    """

    def newMsg(self, name: str, msg) -> None:
        # Ignore frames that we aren't listening for
        if name not in self.streams: return

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs: self.msgs[seq] = dict()
        self.msgs[seq][name] = msg

        if len(self.streams) == len(self.msgs[seq]): # We have sequence num synced frames!

            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            self.queue.put(self.msgs[seq], block=False)

            # Remove previous msgs (memory cleaning)
            for s in self.msgs:
                if int(s) <= int(seq):
                    del self.msgs[s]

# class TimestampSycn(BaseSync):
#     """
#     Timestamp sync will sync all streams based on the timestamp
#     """
#     msgs: Dict[str, List[dai.Buffer]] = dict()  # List of messages
#
#     def newMsg(self, name: str, msg) -> None:
#         # Ignore frames that we aren't listening for
#         if name not in self.streams: return
#         # Return all latest msgs (not synced)
#         if name not in self.msgs: self.msgs[name] = []
#
#         self.msgs[name].append(msg)
#         msgsAvailableCnt = [0 < len(arr) for _, arr in self.msgs.items()].count(True)
#
#         if len(self.streams) == msgsAvailableCnt:
#             # We have at least 1 msg for each stream. Get the latest, remove all others.
#             ret = {}
#             for name, arr in self.msgs.items():
#                 # print(f'len(msgs[{name}])', len(self.msgs[name]))
#                 self.msgs[name] = self.msgs[name][-1:]  # Remove older msgs
#                 # print(f'After removing - len(msgs[{name}])', len(self.msgs[name]))
#                 ret[name] = arr[-1]
#
#             if self.queue.full():
#                 self.queue.get()  # Get one, so queue isn't full
#
#             # print(time.time(),' Putting msg batch into queue. queue size', self.queue.qsize(), 'self.msgs len')
#
#             self.queue.put(ret, block=False)


class TwoStageSeqSync(BaseSync):
    """
    Two stage syncing based on sequence number. Each frame produces ImgDetections msg that contains X detections.
    Each detection (if not on blacklist) will crop the original frame and forward it to the second (stage) NN for
    inferencing.
    """
    labels: Optional[List[int]]
    scaleBbs: Optional[Tuple[int, int]]

    class TwoStageSyncPacket:
        """
        Packet of (two-stage NN) synced messages
        """
        global labels
        global scaleBbs

        frames: List[dai.ImgFrame] = []
        _dets: dai.ImgDetections = None
        recognitions: List[dai.NNData] = []

        @property
        def dets(self) -> dai.ImgDetections:
            return self._dets
        @dets.setter
        def dets(self, dets: dai.ImgDetections):
            # Used to match the scaled bounding boxes by the 2-stage NN script node
            self._dets = dets
            if self.scaleBbs is None: return # No scaling required, ignore

            for det in self._dets.detections:
                # Skip resizing BBs if we have whitelist and the detection label is not on it
                if self.labels and det.label not in self.labels: continue
                det.xmin -= self.scaleBbs[0] / 100
                det.ymin -= self.scaleBbs[1] / 100
                det.xmax += self.scaleBbs[0] / 100
                det.ymax += self.scaleBbs[1] / 100

        def synced(self) -> bool:
            """
            Messages are in sync if:
                - dets is not None
                - We have at least one ImgFrame
                - number of recognition msgs is sufficient
            """
            return (self.dets and 0 < len(self.frames) and  len(self.recognitions) == self._required_recognitions())

        def _required_recognitions(self) -> int:
            """
            Required recognition results for this packet, which depends on number of detections (and white-list labels)
            """
            if self.labels:
                return len([det for det in self.dets.detections if det.label in self.labels])
            else:
                return len(self.dets.detections)


    msgs: Dict[str, TwoStageSyncPacket]
    """
    msgs = {
        '1': TwoStageSyncPacket(),
        '2': TwoStageSyncPacket(),
    }
    """
    frameStreams: List[str]
    detStream: str
    recognitionStream: str

    def __init__(self, callbacks: List[Callable],
                 components: List[Component],
                 frameStreams: List[str],
                 detStream: str,
                 recognitionStream: str,
                 labels = None,
                 scaleBbs = None,
                 ):
        super().__init__(callbacks, components)
        self.labels = labels
        self.scaleBbs = scaleBbs

        self.frameStreams = frameStreams
        self.detStream = detStream
        self.recognitionStream = recognitionStream

    def newMsg(self, name: str, msg) -> None:
        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = self.TwoStageSyncPacket()

        if name == self.recognitionStream:
            self.msgs[seq].recognitions.append(msg)
            # print(f'Added recognition seq {seq}, total len {len(self.msgs[seq]["recognition"])}')
        elif name == self.detStream:
            self.msgs[seq].dets = msg
            # print(f'Added detection seq {seq}')
        elif name in self.frameStreams:
            self.msgs[seq].frames.append(msg)
            # print(f'Added frame seq {seq}')
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if self.msgs[seq].synced():
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            self.queue.put(self.msgs[seq], block=False)

            for s in self.msgs:
                if int(s) <= int(seq):
                    del self.msgs[s]
