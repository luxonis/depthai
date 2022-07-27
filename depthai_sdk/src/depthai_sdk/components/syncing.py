import depthai as dai
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractstaticmethod

class BaseSync(ABC):
    callback: Callable = None

    @abstractstaticmethod
    def newMsg(self, name: str, msg) -> None:
        raise NotImplementedError()

    def setCallback(self, callback: Callable) -> None:
        self.callback = callback

class NoSync(BaseSync):
    """
    Will call callback whenever it gets a new message
    """
    msgs: Dict[str, List[dai.Buffer]] = dict() # List of messages
    streams: Optional[List[str]] = None # Available streams

    def __init__(self, streams: Optional[List[str]] = None) -> None:
        super().__init__()
        self.streams = streams

    def newMsg(self, name: str, msg) -> None:
        if not self.streams:
            self.callback({name: msg})
        else:
            # Return all latest msgs (not synced)
            if name not in self.msgs: self.msgs[name] = []

            self.msgs[name].append(msg)
            msgsAvailableCnt = [0 < len(arr) for name, arr in self.msgs.items()].count(True)

            if len(self.streams) == msgsAvailableCnt:
                # We have at least 1 msg for each stream. Get the latest, remove all others.
                ret = {}
                for name, arr in self.msgs.items():
                    arr = arr[-1:] # Remove older msgs
                    ret[name] = arr[0]

                self.callback(ret)

class SeqSync(BaseSync):
    """
    Will call callback whenever it gets a new message
    """
    msgs: dict = dict() # List of messages

    def newMsg(self, name: str, msg) -> None:
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {} # Create directory for msgs
        
        self.msgs[seq][name] = msg
        self.msgs[name].append(msg)

        if super().callback:
            super().callback(name, msg)

class SeqSycn:
    _msgs = dict()

    def addMsg(self, msg, name: str):
        a = 5
    def getMsgs(self) -> dict:
        seqRemove = [] # Arr of sequence numbers to get deleted
        for seq, msgs in self._msgs.items():
            seqRemove.append(seq) # Will get removed from dict if we find synced msgs pair

            # Check if we have both detections and color frame with this sequence number
            if "detection" in msgs and "color" in msgs:
                # We have synced msgs, remove previous msgs (memory cleaning)
                for rm in seqRemove:
                    del self._msgs[rm]

                return msgs # Returned synced msgs
        return None # No synced msgs


class TwoStageHostSeqSync:
    def __init__(self, labels = None, scaleBbs = None) -> None:
        self.labels = labels
        self.scaleBbs = scaleBbs

    def getSyncedMsgs(self, ogMsgs: Dict[str, List[dai.Buffer]]) -> dict:
        seqRemove = [] # Arr of sequence numbers to get deleted
        seqMsgs = self.SeqArray(ogMsgs)


        for seq, msgs in seqMsgs.items():
            seqRemove.append(seq) # Will get removed from dict if we find synced msgs pair

            if 'detection' not in msgs: return None

            detNum = len([det for det in msgs['detection'].detections if det.label in self.labels])
            # Check if we have both detections and color frame with this sequence number

            print(f"Seq {seq}, det len {detNum}, rec len {len(msgs['recognition'])},  msgs", msgs)
            # Check if all detected objects (faces) have finished recognition inference
            if detNum == len(msgs["recognition"]):
                # print(f"Synced msgs with sequence number {seq}", msgs)

                # We have synced msgs, remove previous msgs (memory cleaning)
                # print(f"\nSYNCED {seq}, removing others\n")
                self.removeSeqNums(ogMsgs, seqRemove)

                if self.scaleBbs:
                    for det in msgs['detection'].detections:
                        if det.label not in self.labels: continue # Only scale whitelist label BBs
                        det.xmin -= self.scaleBbs[0]/100
                        det.ymin -= self.scaleBbs[1]/100
                        det.xmax += self.scaleBbs[0]/100
                        det.ymax += self.scaleBbs[1]/100

                return msgs # Returned synced msgs

        return None # No synced msgs


    def SeqArray(msgsByStream: Dict[str, List[dai.Buffer]]) -> Dict[str, Dict[str, Any]]:
        arr: Dict[str, Dict[str, List[dai.Buffer]]] = dict()
        # Generate dict of sequence numbers, where items are dict of stream names and their msgs
        for name, msgs in msgsByStream.items():
            for msg in msgs:
                seq = str(msg.getSequenceNum())
                if seq not in arr:
                    arr[seq] = {}
                    arr[seq]['recognition'] = []
                # print(f"Stream Name {name}, msgs, seq {seq}")
                # TODO: query from NNComponent
                if name == 'recognition':
                    arr[seq][name].append(msg)
                else:
                    arr[seq][name] = msg
        return arr

    def removeSeqNums(msgs: Dict[str, List[dai.Buffer]], seqRemove: List[str]) -> None:
        for _, msgs in msgs.items():
            for msg in msgs:
                if str(msg.getSequenceNum()) in seqRemove:
                    msgs.remove(msg)