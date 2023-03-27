from typing import Dict, List, Any, Optional
import itertools
from collections import deque

class MessageSync:
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

    #FIXME Get these parameters from somewhere, using these defaults, especially for min_diff_timestamp is not a good idea
    def __init__(self, stream_num: int, type: str = 'ts', min_diff_timestamp=10, max_num_messages=10, min_queue_depth=2):
        self.type = type
        self.msgs: Dict[str, Dict[str, Any]] = dict()
        self.num_queues = stream_num
        self.min_diff_timestamp = min_diff_timestamp
        self.max_num_messages = max_num_messages
        self.queues = dict()
        self.queue_depth = min_queue_depth

        self.stream_num: int = stream_num
        if self.type == 'seq':
            self.sync = self.seq_sync
        elif self.type == 'ts':
            self.sync = self.ts_sync
        else:
            raise RuntimeError(f"Sync type '{self.type}' not recognized! Should be either 'seq' or 'ts'")


    def seq_sync(self, seq_num: int, name: str, msg) -> Optional[Dict]:
        seq_num = str(seq_num)
        if seq_num not in self.msgs: self.msgs[seq_num] = dict()

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


    def ts_sync(self, seq_num: int, name: str, msg): # Compute intensive implementation, check for a better way
        if name not in self.queues:
            self.queues[name] = deque(maxlen=self.max_num_messages)
        self.queues[name].append(msg)
        # Atleast 3 messages should be buffered
        min_len = min([len(queue) for queue in self.queues.values()])
        if min_len == 0:
            # print('Status:', 'exited due to min len == 0', self.queues)
            return None

        # initializing list of list
        queue_lengths = []
        for name in self.queues.keys():
            queue_lengths.append(range(0, len(self.queues[name])))
        permutations = list(itertools.product(*queue_lengths))
        # print ("All possible permutations are : " +  str(permutations))

        # Return a best combination after being atleast 3 messages deep for all queues
        min_ts_diff = None
        for indicies in permutations:
            tmp = {}
            i = 0
            for n in self.queues.keys():
                tmp[n] = indicies[i]
                i = i + 1
            indicies = tmp

            acc_diff = 0.0
            min_ts = None
            for name in indicies.keys():
                msg = self.queues[name][indicies[name]]
                if min_ts is None:
                    min_ts = msg.getTimestampDevice().total_seconds()
            for name in indicies.keys():
                msg = self.queues[name][indicies[name]]
                acc_diff = acc_diff + abs(min_ts - msg.getTimestampDevice().total_seconds())

            # Mark minimum
            if min_ts_diff is None or (acc_diff < min_ts_diff['ts'] and abs(acc_diff - min_ts_diff['ts']) > 0.0001):
                min_ts_diff = {'ts': acc_diff, 'indicies': indicies.copy()}
                # print('new minimum:', min_ts_diff, 'min required:', self.min_diff_timestamp)

            if min_ts_diff['ts'] < self.min_diff_timestamp:
                # Check if atleast 5 messages deep
                min_queue_depth = None
                for name in indicies.keys():
                    if min_queue_depth is None or indicies[name] < min_queue_depth:
                        min_queue_depth = indicies[name]
                if min_queue_depth >= self.queue_depth:
                    # Retrieve and pop the others
                    synced = {}
                    for name in indicies.keys():
                        synced[name] = self.queues[name][min_ts_diff['indicies'][name]]
                        # pop out the older messages
                        for i in range(0, min_ts_diff['indicies'][name]+1):
                            self.queues[name].popleft()

                    # print('Returning synced messages with error:', min_ts_diff['ts'], min_ts_diff['indicies'])
                    return synced

