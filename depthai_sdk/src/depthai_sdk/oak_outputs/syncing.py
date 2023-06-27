import threading
from typing import Dict, List, Any, Optional


class SequenceNumSync:
    """
        msgs = {seqNum: {name: message}}
        Example:

        msgs = {
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
