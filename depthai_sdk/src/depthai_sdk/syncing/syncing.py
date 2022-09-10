import depthai as dai
from typing import Dict, List, Any, Optional, Tuple, Callable
from queue import Empty, Queue
from abc import ABC, abstractmethod

from ..components.component_group import ComponentGroup
from ..components import Component


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
            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seq):
                    newMsgs[name] = msg
            self.msgs = newMsgs

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
    labels: Optional[List[int]] = None
    scaleBbs: Optional[Tuple[int, int]] = None

    msgs: Dict[str, Dict[str, Any]] = dict() # List of messages
    """
    msgs = {
        '1': TwoStageSyncPacket(),
        '2': TwoStageSyncPacket(), 
    }
    """
    group: ComponentGroup
    def __init__(self, callbacks: List[Callable],
                 group
                 ):
        super().__init__(callbacks, group.components)

        if group.second_nn.multi_stage_config:
            self.scaleBbs = group.second_nn.multi_stage_config.scaleBb
            self.labels = group.second_nn.multi_stage_config.labels

        self.group = group

    def newMsg(self, name: str, msg: dai.Buffer) -> None:
        if name not in self.streams: return # From Replay modules. TODO: better handling?

        # TODO: what if msg doesn't have sequence num?
        seq = str(msg.getSequenceNum())

        if seq not in self.msgs:
            self.msgs[seq] = dict()
            self.msgs[seq][self.group.second_nn_name] = []
            self.msgs[seq][self.group.nn_name] = None

        if name == self.group.second_nn_name:
            self.msgs[seq][name].append(msg)
            # print(f'Added recognition seq {seq}, total len {len(self.msgs[seq]["recognition"])}')
        elif name == self.group.nn_name:
            self.add_detections(seq, msg)
            # print(f'Added detection seq {seq}')
        elif name in self.group.frame_names:
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

    def add_detections(self, seq:str , dets: dai.ImgDetections):
        # Used to match the scaled bounding boxes by the 2-stage NN script node
        self.msgs[seq][self.group.nn_name] = dets

        if self.scaleBbs is None: return  # No scaling required, ignore

        for det in dets.detections:
            # Skip resizing BBs if we have whitelist and the detection label is not on it
            if self.labels and det.label not in self.labels: continue
            det.xmin -= self.scaleBbs[0] / 100
            det.ymin -= self.scaleBbs[1] / 100
            det.xmax += self.scaleBbs[0] / 100
            det.ymax += self.scaleBbs[1] / 100

    def synced(self, seq: str) -> bool:
        """
        Messages are in sync if:
            - dets is not None
            - We have at least one ImgFrame
            - number of recognition msgs is sufficient
        """
        packet = self.msgs[seq]

        for name in self.group.frame_names:
            # print(f'checking if stream {name} in {self.group.frame_names}')
            if name not in packet:
                return False  # We don't have required ImgFrame

        if not packet[self.group.nn_name]:
            return False  # We don't have dai.ImgDetections

        if len(packet[self.group.second_nn_name]) < self.required_recognitions(seq):
            return False  # We don't have enough 2nd stage NN results
        return True

    def required_recognitions(self, seq: str) -> int:
        """
        Required recognition results for this packet, which depends on number of detections (and white-list labels)
        """
        dets: List[dai.ImgDetection] = self.msgs[seq][self.group.nn_name].detections
        print('required_recognitions, dets', dets)
        if self.labels:
            return len([det for det in dets if det.label in self.labels])
        else:
            return len(dets)
