from queue import Queue
from threading import Thread
from typing import Dict, Any

import rclpy

from depthai_sdk.integrations.ros.ros_base import RosBase
from depthai_sdk.logger import LOGGER


def ros_thread(queue: Queue):
    rclpy.init()
    node = rclpy.create_node('DepthAI_SDK')
    publishers = dict()

    while rclpy.ok():
        msgs: Dict[str, Any] = queue.get(block=True)
        for topic, msg in msgs.items():
            if topic not in publishers:
                publishers[topic] = node.create_publisher(type(msg), topic, 10)
                LOGGER.info(f'SDK started publishing ROS messages to {topic}')
            publishers[topic].publish(msg)
        rclpy.spin_once(node, timeout_sec=0.001)  # 1ms timeout


class Ros2Streaming(RosBase):
    queue: Queue

    def __init__(self):
        self.queue = Queue(30)
        self.process = Thread(target=ros_thread, args=(self.queue,))
        self.process.start()
        super().__init__()

    # def update(self): # By RosBase
    # def new_msg(self): # By RosBase

    def new_ros_msg(self, topic: str, ros_msg):
        self.queue.put({topic: ros_msg})
