from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager, NNetManager, BlobManager
import depthai as dai
import cv2

print("This test is made to test nnet manager, but it also contains blob, preview and pipeline manager.\n"
      "If the face detection is running, then that means that NNet manager works,"
      " it also confirms that getBlob from Blob manager works.\n")
pm = PipelineManager()
pm.createColorCam(xout=True)

bm = BlobManager(zooName="face-detection-retail-0004")

nm = NNetManager(inputSize=(300, 300), nnFamily="mobilenet", labels=["bg", "face"])
nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion()))
pm.addNn(nn)

with dai.Device(pm.pipeline) as device:
    pv = PreviewManager(display=[Previews.color.name])
    pv.createQueues(device)

    nm.createQueues(device)
    nnData = []

    while True:
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            nnData = nm.decode(inNn)

        nm.draw(pv, nnData)
        pv.showFrames()

        if cv2.waitKey(1) == ord('q'):
            break
pv.closeQueues()
nm.closeQueues()
