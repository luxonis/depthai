from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DisparityDepthPacket
import cv2
from depthai_sdk.visualize.visualizer import Visualizer

with OakCamera() as oak:
    cam_c = oak.create_camera('CAM_C')
    tof = oak.create_tof("CAM_A", align_to=cam_c)
    depth_q = oak.queue(tof.out.depth).queue

    vis = Visualizer() # Only for depth colorization
    oak.start()
    while oak.running():
        depth: DisparityDepthPacket = depth_q.get()
        colored_depth = depth.get_colorized_frame(vis)
        cv2.imshow("depth", colored_depth)
        cv2.imshow('Weighted', cv2.addWeighted(depth.aligned_frame.getCvFrame(), 0.5, colored_depth, 0.5, 0))
        if cv2.waitKey(1) == ord('q'):
            break
