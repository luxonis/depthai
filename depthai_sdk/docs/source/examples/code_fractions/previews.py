from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager
import depthai as dai

# create pipeline
pm = PipelineManager()

# creating color source
pm.createColorCam(xout=True)

# connecting to the device
with dai.Device(pm.pipeline) as device:
    # define configs for above sources
    pv = PreviewManager(display=[Previews.color.name])

    # create stream queues
    pv.createQueues(device)

    # prepare and show streams
    pv.prepareFrames()
    pv.showFrames()