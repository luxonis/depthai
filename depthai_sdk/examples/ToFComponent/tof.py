from depthai_sdk import OakCamera

with OakCamera() as oak:
    tof = oak.create_tof("cama")
    depth_q = oak.queue(tof.out.depth).queue
    amplitude_q = oak.queue(tof.out.amplitude).queue
    oak.visualize([tof.out.depth, tof.out.amplitude])
    oak.start(blocking=True)
