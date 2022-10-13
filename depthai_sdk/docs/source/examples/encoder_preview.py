from pathlib import Path
from depthai_sdk import EncodingManager, Previews
from depthai_sdk.managers import PipelineManager
import depthai as dai


# first create a dictionary with wanted streams as keys and fps number with their values
encodeConfig = dict()
encodeConfig[Previews.color.name] = 30

# create encoder with above declared dictionary and path to save the file ("" will save it next to the program file)
em = EncodingManager(encodeConfig, Path(__file__))

# create pipeline with above mentioned streams
pm = PipelineManager()
pm.createColorCam(xoutVideo=True, previewSize=(300, 300))

# create encoders for all streams that were initialized
em.createEncoders(pm)

# start device
with dai.Device(pm.pipeline) as device:
    # create stream queues
    em.createDefaultQueues(device)

    while True:
        # save frames to .h files
        em.parseQueues()