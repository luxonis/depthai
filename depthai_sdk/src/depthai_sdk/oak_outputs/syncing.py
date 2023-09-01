import threading
from datetime import timedelta
from typing import Dict, Any, Optional


class SequenceNumSync:
    """
        self.msgs = {seqNum: {name: message}}
        Example:

        self.msgs = {
            '1': {
                'rgb': dai.Frame(),
                'dets': dai.ImgDetections(),
            ],
            '2': {
                'rgb': dai.Frame(),
                'dets': dai.ImgDetections(),
            }
        }
        """

    def __init__(self, stream_num: int):
        self.msgs: Dict[str, Dict[str, Any]] = dict()
        self.stream_num: int = stream_num
        self.lock = threading.Lock()

    def sync(self, seq_num: int, name: str, msg) -> Optional[Dict]:
        seq_num = str(seq_num)

        with self.lock:
            if seq_num not in self.msgs:
                self.msgs[seq_num] = dict()

            self.msgs[seq_num][name] = (msg)

            if self.stream_num == len(self.msgs[seq_num]):
                # We have sequence num synced frames!
                ret = self.msgs[seq_num]

                # Remove previous msgs
                new_msgs = {}
                for name, msg in self.msgs.items():
                    if int(name) > int(seq_num):
                        new_msgs[name] = msg
                self.msgs = new_msgs

                return ret

        return None


class TimestampSync:
    def __init__(self, stream_num: int, ms_threshold: int):
        self.msgs: Dict[str, Any] = dict()
        self.stream_num: int = stream_num
        self.ms_threshold = ms_threshold

    def sync(self, timestamp, name: str, msg):
        if name not in self.msgs:
            self.msgs[name] = []

        self.msgs[name].append((timestamp, msg))

        synced = {}
        for name, arr in self.msgs.items():
            # Go through all stored messages and calculate the time difference to the target msg.
            # Then sort these self.msgs to find a msg that's closest to the target time, and check
            # whether it's below 17ms which is considered in-sync.
            diffs = []
            for i, (msg_ts, msg) in enumerate(arr):
                diffs.append(abs(msg_ts - timestamp))
            if len(diffs) == 0:
                break
            diffs_sorted = diffs.copy()
            diffs_sorted.sort()
            dif = diffs_sorted[0]

            if dif < timedelta(milliseconds=self.ms_threshold):
                synced[name] = diffs.index(dif)

        if len(synced) == self.stream_num:  # We have all synced streams
            # Remove older self.msgs
            for name, i in synced.items():
                self.msgs[name] = self.msgs[name][i:]
            ret = {}
            for name, arr in self.msgs.items():
                ts, synced_msg = arr.pop(0)
                ret[name] = synced_msg
            return ret
        return None
