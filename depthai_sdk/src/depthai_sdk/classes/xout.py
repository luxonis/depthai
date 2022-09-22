import depthai as dai
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from .xout_base import XoutBase, StreamXout

"""
Xout classes are abstracting streaming messages to the host computer (via XLinkOut) and syncing those messages
on the host side before sending (synced) messages to message sinks (eg. visualizers, or loggers).
TODO:
- separate syncing logic from the class. XoutTwoStage should extend the XoutNnResults (currently can't as syncing logic is not separated)
"""

class XoutFrames(XoutBase):
    """
    Single message, no syncing required
    """
    frames: StreamXout
    def __init__(self, cb: Callable, frames: StreamXout):
        self.frames = frames
        super().__init__()
        self.setup_base(cb)

    def xstreams(self) -> List[StreamXout]:
        return [self.frames]

    def newMsg(self, name: str, msg) -> None:
        # Ignore frames that we aren't listening for
        if name not in self._streams: return

        if self.queue.full():
            self.queue.get()  # Get one, so queue isn't full

        self.queue.put({name: msg}, block=False)


class XoutSequenceSync(XoutBase):
    """
    This class will sync all messages based on their sequence number
    """
    msgs: Dict[str, Dict[str, dai.Buffer]]  # List of messages.

    def __init__(self):
        super().__init__()
        self.msgs = dict()

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
        if name not in self._streams: return

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs: self.msgs[seq] = dict()
        self.msgs[seq][name] = msg

        if len(self._streams) == len(self.msgs[seq]):  # We have sequence num synced frames!

            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            self.queue.put(self.msgs[seq], block=False)

            # Remove previous msgs (memory cleaning)
            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs


class XoutSpatialBoundingBox(XoutSequenceSync):
    frames: StreamXout
    bb_stream: StreamXout

    def __init__(self, nnComp, cb: Callable, frames: StreamXout, bb_stream: StreamXout):
        self.frames = frames
        self.bb_stream = bb_stream
        # Save StreamXout before initializing super()!
        super().__init__()
        self.setup_base(cb)
        self.nnComp = nnComp

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.bb_stream]


class XoutNnResults(XoutSequenceSync):
    frames: StreamXout
    nn_results: StreamXout

    def __init__(self, nnComp, cb: Callable, frames: StreamXout, nn_results: StreamXout):
        self.frames = frames
        self.nn_results = nn_results
        # Save StreamXout before initializing super()!
        super().__init__()
        self.setup_base(cb)
        self.nnComp = nnComp

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results]


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


class XoutTwoStage(XoutBase):
    """
    Two stage syncing based on sequence number. Each frame produces ImgDetections msg that contains X detections.
    Each detection (if not on blacklist) will crop the original frame and forward it to the second (stage) NN for
    inferencing.
    """
    msgs: Dict[str, Dict[str, Any]] = dict()  # List of messages
    """
    msgs = {
        '1': TwoStageSyncPacket(),
        '2': TwoStageSyncPacket(), 
    }
    """
    labels: Optional[List[int]] = None
    scaleBb: Optional[Tuple[int, int]] = None

    frames: StreamXout
    nn_results: StreamXout
    second_nn: StreamXout

    def __init__(self, detectionComp, cb: Callable, frames: StreamXout, detections: StreamXout, second_nn: StreamXout):
        self.frames = frames
        self.nn_results = detections
        self.second_nn = second_nn
        # Save StreamXout before initializing super()!
        super().__init__()
        self.setup_base(cb)

        self.nnComp = detectionComp

        conf = detectionComp._multi_stage_config # No types due to circular import...
        if conf is not None:
            self.labels = conf.labels
            self.scaleBb = conf.scaleBb

    def xstreams(self) -> List[StreamXout]:
        return [self.frames, self.nn_results, self.second_nn]

    def newMsg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self._streams: return  # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()
            self.msgs[seq][self.second_nn.name] = []
            self.msgs[seq][self.nn_results.name] = None

        if name == self.second_nn.name:
            self.msgs[seq][name].append(msg)
            # print(f'Added recognition seq {seq}, total len {len(self.msgs[seq]["recognition"])}')
        elif name == self.nn_results.name:
            self.add_detections(seq, msg)
            # print(f'Added detection seq {seq}')
        elif name in self.frames.name:
            self.msgs[seq][name] = msg
            # print(f'Added frame seq {seq}')
        else:
            raise ValueError('Message from unknown stream name received by TwoStageSeqSync!')

        if self.synced(seq):
            # Frames synced!
            if self.queue.full():
                self.queue.get()  # Get one, so queue isn't full

            # for name in self.frame_names:
            #     packet = dict()
            #     packet[self.]

            self.queue.put(self.msgs[seq], block=False)

            # Throws RuntimeError: dictionary changed size during iteration
            # for s in self.msgs:
            #     if int(s) <= int(seq):
            #         del self.msgs[s]

            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs

    def add_detections(self, seq: str, dets: dai.ImgDetections):
        # Used to match the scaled bounding boxes by the 2-stage NN script node
        self.msgs[seq][self.nn_results.name] = dets

        if self.scaleBb is None: return  # No scaling required, ignore

        for det in dets.detections:
            # Skip resizing BBs if we have whitelist and the detection label is not on it
            if self.labels and det.label not in self.labels: continue
            det.xmin -= self.scaleBb[0] / 100
            det.ymin -= self.scaleBb[1] / 100
            det.xmax += self.scaleBb[0] / 100
            det.ymax += self.scaleBb[1] / 100

    def synced(self, seq: str) -> bool:
        """
        Messages are in sync if:
            - dets is not None
            - We have at least one ImgFrame
            - number of recognition msgs is sufficient
        """
        packet = self.msgs[seq]

        for name in self.frames.name:
            # print(f'checking if stream {name} in {self.group.frame_names}')
            if name not in packet:
                return False  # We don't have required ImgFrames

        if not packet[self.nn_results.name]:
            return False  # We don't have dai.ImgDetections

        if len(packet[self.second_nn.name]) < self.required_recognitions(seq):
            return False  # We don't have enough 2nd stage NN results
        return True

    def required_recognitions(self, seq: str) -> int:
        """
        Required recognition results for this packet, which depends on number of detections (and white-list labels)
        """
        dets: List[dai.ImgDetection] = self.msgs[seq][self.nn_results.name].nn_results
        if self.labels:
            return len([det for det in dets if det.label in self.labels])
        else:
            return len(dets)
