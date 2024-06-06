from depthai_sdk import OakCamera
from depthai_sdk.visualize.configs import StereoColor
from depthai_sdk.classes.packets import DisparityDepthPacket
from depthai_sdk.visualize.visualizers.opencv_visualizer import OpenCvVisualizer
import math
import depthai as dai
import cv2

# User-defined constants
WARNING = 1000 # 1m, orange
CRITICAL = 500 # 50cm, red

slc_data = []
fontType = cv2.FONT_HERSHEY_TRIPLEX

    

with OakCamera() as oak:
    stereo = oak.create_stereo('720p')
    # We don't need high fill rate, just very accurate depth, that's why we enable some filters, and
    # set the confidence threshold to 50
    config = stereo.node.initialConfig.get()
    config.postProcessing.brightnessFilter.minBrightness = 0
    config.postProcessing.brightnessFilter.maxBrightness = 255
    stereo.node.initialConfig.set(config)
    stereo.config_postprocessing(colorize=StereoColor.RGBD, colormap=cv2.COLORMAP_BONE)
    stereo.config_stereo(confidence=50, lr_check=True, extended=True)


    slc = oak.pipeline.create(dai.node.SpatialLocationCalculator)
    for x in range(15):
        for y in range(9):
            config = dai.SpatialLocationCalculatorConfigData()
            config.depthThresholds.lowerThreshold = 200
            config.depthThresholds.upperThreshold = 10000
            config.roi = dai.Rect(dai.Point2f((x+0.5)*0.0625, (y+0.5)*0.1), dai.Point2f((x+1.5)*0.0625, (y+1.5)*0.1))
            # TODO: change from median to 10th percentile once supported
            config.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
            slc.initialConfig.addROI(config)

    stereo.depth.link(slc.inputDepth)

    slc_out = oak.pipeline.create(dai.node.XLinkOut)
    slc_out.setStreamName('slc')
    slc.out.link(slc_out.input)

    stereoQ = oak.queue(stereo.out.depth).get_queue()

    oak.start() # Start the pipeline (upload it to the OAK)

    slcQ = oak.device.getOutputQueue('slc') # Create output queue after calling start()
    vis = OpenCvVisualizer()
    while oak.running():
        oak.poll()
        packet: DisparityDepthPacket = stereoQ.get()
        slc_data = slcQ.get().getSpatialLocations()

        depthFrameColor = packet.get_colorized_frame(vis)

        for depthData in slc_data:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])

            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            coords = depthData.spatialCoordinates
            distance = math.sqrt(coords.x ** 2 + coords.y ** 2 + coords.z ** 2)

            if distance == 0: # Invalid
                continue

            if distance < CRITICAL:
                color = (0, 0, 255)
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=4)
                cv2.putText(depthFrameColor, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), fontType, 0.5, color)
            elif distance < WARNING:
                color = (0, 140, 255)
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, thickness=2)
                cv2.putText(depthFrameColor, "{:.1f}m".format(distance/1000), (xmin + 10, ymin + 20), fontType, 0.5, color)

        cv2.imshow('Frame', depthFrameColor)
