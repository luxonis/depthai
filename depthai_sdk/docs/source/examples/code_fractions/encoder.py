from pathlib import Path
from depthai_sdk import EncodingManager
from depthai_sdk.managers import PipelineManager
import depthai as dai


# Before declaring the pipeline and encoder, we must create a dictionary of streams, with fps ans values
encodeConfig = dict()
encodeConfig["color"] = 30

# create encoder with above declared dictionary and path to save the file ("" will save it next to the program file)
em = EncodingManager(encodeConfig, Path(""))

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
        # save frames to .h256 files
        em.parseQueues()