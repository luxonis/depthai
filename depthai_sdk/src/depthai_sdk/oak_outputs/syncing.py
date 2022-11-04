from typing import Dict, List, Any, Optional


class SequenceNumSync:
    msgs: Dict[str, Dict[str, Any]]  # List of messages.
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
    streamNum: int

    def __init__(self, streamNum: int):
        self.msgs = dict()
        self.streamNum = streamNum

    def sync(self, seqNum: int, name: str, msg) -> Optional[Dict]:
        seqNum = str(seqNum)
        if seqNum not in self.msgs: self.msgs[seqNum] = dict()

        self.msgs[seqNum][name] = (msg)

        if self.streamNum == len(self.msgs[seqNum]):
            # We have sequence num synced frames!
            ret = self.msgs[seqNum]

            # Remove previous msgs
            newMsgs = {}
            for name, msg in self.msgs.items():
                if int(name) > int(seqNum):
                    newMsgs[name] = msg
            self.msgs = newMsgs

            return ret
        return None
