from depthai_sdk import OakCamera, TwoStagePacket, AspectRatioResizeMode, Visualizer
import depthai as dai
import numpy as np
import cv2

with OakCamera() as oak:
    color = oak.create_camera('color')

    det = oak.create_nn('face-detection-retail-0004', color)
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det.config_nn(aspectRatioResizeMode=AspectRatioResizeMode.CROP)

    age_gender = oak.create_nn('age-gender-recognition-retail-0013', input=det)
    # age_gender.config_multistage_nn(show_cropped_frames=True) # For debugging

    def cb(packet: TwoStagePacket):
        for det in packet.detections:
            rec = det.nn_data
            age = int(float(np.squeeze(np.array(rec.getLayerFp16('age_conv3')))) * 100)
            gender = np.squeeze(np.array(rec.getLayerFp16('prob')))
            gender_str = "woman" if gender[0] > gender[1] else "man"
            Visualizer.putText(packet.frame, f"{gender_str}, age: {age}", (det.topLeft[0]+5, det.topLeft[1]+45), scale=0.8)

        cv2.imshow(packet.name, packet.frame)


    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(age_gender, fps=True, callback=cb)

    # oak.show_graph() # Show pipeline graph, no need for now
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)