from depthai_sdk import OakCamera
from depthai_sdk.classes.packets import DetectionPacket
import depthai as dai
import cv2

# We use callback, so we only have cv2 window for both models
def cb(packet: DetectionPacket):
    frame = packet.visualizer.draw(packet.frame)
    cv2.imshow('Frame', frame)

with OakCamera() as oak:
    color = oak.create_camera('color')

    script = oak.pipeline.create(dai.node.Script)
    # When Script node receives a message from the host it will switch streaming frames to another NN model
    script.setScript("""
    i = 0
    outputs = ['out1', 'out2']

    while True:
        frame = node.io['frames'].get()

        switch = node.io['switch'].tryGet()
        if switch is not None:
            i += 1
            if len(outputs) <= i:
                i = 0

        node.io[outputs[i]].send(frame)
    """)
    color.stream.link(script.inputs['frames'])

    # We can have multiple models here, not just 2 object detection models
    nn1 = oak.create_nn('yolov6nr3_coco_640x352', input=script.outputs['out1'])
    nn1.config_nn(resize_mode='stretch') # otherwise, BB mappings will be incorrect
    nn2 = oak.create_nn('mobilenet-ssd', input=script.outputs['out2'])
    nn2.config_nn(resize_mode='stretch') # otherwise, BB mappings will be incorrect

    # We will send "switch" message via XLinkIn
    xin = oak.pipeline.create(dai.node.XLinkIn)
    xin.setStreamName('switch')
    xin.out.link(script.inputs['switch'])

    # We don't want syncing, we just want either of the model packets in the callback
    oak.visualize([nn1, nn2], fps=True, callback=cb)

    oak.visualize([nn1.out.passthrough, nn2.out.passthrough], fps=True)

    # oak.show_graph()

    oak.start()
    qin = oak.device.getInputQueue('switch')

    while True:
        key = oak.poll()
        if key == ord('s'):
            print('Switching NN model')
            qin.send(dai.Buffer())
        elif key == ord('q'):
            break
