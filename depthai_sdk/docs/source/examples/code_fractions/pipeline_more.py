from depthai_sdk.managers import PipelineManager
import depthai as dai


# declaring the pipeline
pm = PipelineManager()

# after the declaration, we define it's sources

# color camera
pm.createColorCam(xout=True, previewSize=(500, 500))

# left camera
pm.createLeftCam(xout=True)

# right camera
pm.createRightCam(xout=True)

# depth
pm.createDepth(useDepth=True)

# connecting to device
with dai.Device(pm.pipeline) as device:
    # pipeline is created and the device is connected
    print("Successfully connected to the device!")