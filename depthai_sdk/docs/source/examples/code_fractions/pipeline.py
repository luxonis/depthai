from depthai_sdk.managers import PipelineManager
import depthai as dai


# declaring the pipeline
pm = PipelineManager()

# after the declaration, we define it's sources
pm.createColorCam(xout=True)

# connecting to device
with dai.Device(pm.pipeline) as device:
    # pipeline is created and the device is connected
    print("Successfully connected to the device!")