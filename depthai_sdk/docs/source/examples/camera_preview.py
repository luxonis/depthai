from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager
import depthai as dai
import cv2


# create pipeline
pm = PipelineManager()

# define sources (color, left, right, depth)

# creating color source
pm.createColorCam(xout=True)
pm.createLeftCam(xout=True)
pm.createRightCam(xout=True)
pm.createDepth(useDepth=True)

# connecting to the device
with dai.Device(pm.pipeline) as device:
    # define configs for above sources
    pv = PreviewManager(display=[Previews.color.name, Previews.left.name, Previews.right.name, Previews.depth.name])

    # create stream queues
    pv.createQueues(device)

    while True:
        # prepare and show streams
        pv.prepareFrames()
        pv.showFrames()

        # end program with 'q'
        if cv2.waitKey(1) == ord('q'):
            break