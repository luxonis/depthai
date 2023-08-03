from depthai_sdk import OakCamera
import depthai as dai

with OakCamera() as oak:
    color = oak.create_camera('color')
    nn = oak.create_nn('mobilenet-ssd', color)
    oak.visualize([nn.out.passthrough, nn], fps=True)

    nn.node.setNumInferenceThreads(2) # Configure components' nodes

    features = oak.pipeline.create(dai.node.FeatureTracker) # Create new pipeline nodes
    color.node.video.link(features.inputImage)

    out = oak.pipeline.create(dai.node.XLinkOut)
    out.setStreamName('features')
    features.outputFeatures.link(out.input)

    oak.start() # Start the pipeline (upload it to the OAK)

    q = oak.device.getOutputQueue('features') # Create output queue after calling start()
    while oak.running():
        if q.has():
            result = q.get()
            print(result)
        # Since we are not in blocking mode, we have to poll oak camera to
        # visualize frames, call callbacks, process keyboard keys, etc.
        oak.poll()