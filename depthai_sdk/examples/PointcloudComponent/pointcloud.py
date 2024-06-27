from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.camera('color')
    stereo = oak.create_stereo()
    stereo.config_stereo(align=color)
    pcl = oak.create_pointcloud(depth_input=stereo, colorize=color)
    oak.visualize(pcl, visualizer='depthai-viewer')
    oak.start(blocking=True)
