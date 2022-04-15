from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager
import depthai as dai
import cv2


print("This test is meant to test the Preview manager, but it also contains Pipeline manager.\n"
      "If everything works correctly you should see 3 frames (left, right and color).\n")

pm = PipelineManager()

pm.createColorCam(xout=True)
pm.createLeftCam(xout=True)
pm.createRightCam(xout=True)

with dai.Device(pm.pipeline) as device:
    pv = PreviewManager(display=[Previews.color.name, Previews.left.name, Previews.right.name])
    pv.createQueues(device)

    while True:
        pv.prepareFrames()
        pv.showFrames()
        if cv2.waitKey(1) == ord('q'):
            break

pv.closeQueues()
