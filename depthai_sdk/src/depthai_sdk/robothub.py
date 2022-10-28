import atexit
import signal
import time
from threading import Event, Thread, Condition
from typing import Callable

from depthai_sdk.visualize.visualizer import Visualizer
from depthai_sdk.classes.packets import FramePacket

IS_ROBOTHUB = False


class RobotHub:
    """Auxiliary class that provides functionality for the DepthAI SDK to interact with the RobotHub SDK."""

    def __init__(self, oak_camera: 'OakCamera'):
        self.oak_camera = oak_camera
        self.mxid = None
        self.app: 'App' = None
        self.report_stats_every: int = 10
        self.report_stats_callback: Callable = self._device_stats_callback

        self.running = True

        self.stop_event = Event()
        self.timeout = Condition()
        self.report_info_thread = Thread(target=self.report_device_info, daemon=False)

        atexit.register(self._stop_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def set_app(self, app: 'App'):
        self.app = app

    def report_device_info(self):
        while not self.stop_event.is_set():
            self.report_stats_callback()
            time.sleep(self.report_stats_every)

    def _device_stats_callback(self):
        # self.app.agent_client.publish_device_info({
        #     'serialNumber': self.oak_camera.device,
        #     'name': self.oak_camera.device,
        #     'type': 'OAK-D',
        #     'firmwareVersion': '0.0.0',
        #     'hardwareVersion': '0.0.0',
        #     'connected': True
        # })
        print('device stats callback')

    def add_stream(self, unique_key: str, description: str):
        return self.app.add_stream(self.mxid, unique_key, description)

    def stream_callback(self, stream: 'VideoStream', packet: FramePacket, visualizer: Visualizer):
        if stream.active:
            self.app.agent_client.publish_stream_data(packet.imgFrame.getData().tobytes(), stream.metadata)
        # print(visualizer.serialize())

    def configure(self, report_stats_every: int = 10):
        self.report_stats_every = report_stats_every

    def start(self):
        if not self.oak_camera._pipeline_built:
            self.oak_camera.build()  # Build the pipeline

        self.oak_camera._oak.device.startPipeline(self.oak_camera._pipeline)
        self.oak_camera._oak.initCallbacks(self.oak_camera._pipeline)

        self.mxid = self.oak_camera.device.getMxId()

        for xout in self.oak_camera._oak.oak_out_streams:  # Start FPS counters
            xout.start_fps()

        self.app.agent_client.publish_device_info({
            'serialNumber': self.mxid,
            'status': 'connected',
        })

        self.report_info_thread.start()

        print('RobotHub started')
        # Constant loop: get messages, call callbacks
        while self.running:
            time.sleep(0.001)
            if not self.oak_camera.poll():
                self.app.agent_client.publish_device_info({'serialNumber': self.mxid, 'status': 'disconnected'})
                self.stop()
                break

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.report_info_thread.is_alive():
            self.report_info_thread.join()
        print('RobotHub stopped')

    def _signal_handler(self, unused_signum, unused_frame) -> None:
        atexit.unregister(self._stop_handler)
        self._stop_handler()

    def _stop_handler(self) -> None:
        self.stop()
