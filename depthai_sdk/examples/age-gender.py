from depthai_sdk import OakCamera, TwoStagePacket, AspectRatioResizeMode, Visualizer
import depthai as dai
import numpy as np
import cv2

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

with OakCamera() as oak:
    color = oak.create_camera('color', out='color')

    det = oak.create_nn('face-detection-retail-0004', color, out='dets')
    # Passthrough is enabled for debugging purposes
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det.config_nn(passthroughOut=True, aspectRatioResizeMode=AspectRatioResizeMode.CROP)

    emotion_nn = oak.create_nn('emotions-recognition-retail-0003', input=det, out='reid')
    emotion_nn.config_multistage_nn(show_cropped_frames=True) # For debugging

    def cb(packet: TwoStagePacket):
        for det in packet.detections:
            emotion_results = np.array(det.nn_data.getFirstLayerFp16())
            emotion_name = emotions[np.argmax(emotion_results)]
            Visualizer.putText(packet.frame, emotion_name, (det.topLeft[0]+5, det.topLeft[1]+45), scale=0.8)
        cv2.imshow(packet.name, packet.frame)


    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize([color, det, emotion_nn], fps=True, callback=cb)
    # oak.show_graph() # Show pipeline graph, no need for now
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)
