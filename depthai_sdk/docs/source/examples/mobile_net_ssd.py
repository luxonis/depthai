from depthai_sdk import Previews
from depthai_sdk.managers import PipelineManager, PreviewManager, NNetManager, BlobManager
import depthai as dai
import cv2

# create pipeline
pm = PipelineManager()

# define camera source (in this case color and change winow size to 600 x 500)
pm.createColorCam(xout=True, previewSize=(600, 500))

# define project that you wish to run
bm = BlobManager(zooName="mobilenet-ssd")

# define Neural network configs
nm = NNetManager(inputSize=(300, 300), nnFamily="mobilenet")
nn = nm.createNN(pipeline=pm.pipeline, nodes=pm.nodes, source=Previews.color.name,
                 blobPath=bm.getBlob(shaves=6, openvinoVersion=pm.pipeline.getOpenVINOVersion()))
pm.addNn(nn)

# connect to device
with dai.Device(pm.pipeline) as device:
    # define configs for above sources
    pv = PreviewManager(display=[Previews.color.name])

    # create stream and neural network queues
    pv.createQueues(device)
    nm.createQueues(device)
    nnData = []

    while True:
        # read frames
        pv.prepareFrames()
        inNn = nm.outputQueue.tryGet()

        if inNn is not None:
            nnData = nm.decode(inNn)

        # draw information on frame and show frame
        nm.draw(pv, nnData)
        pv.showFrames()

        # end program with 'q'
        if cv2.waitKey(1) == ord('q'):
            break