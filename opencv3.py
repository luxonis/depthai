from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

if __name__ == 'main':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="redsquare-gwdyn", confidence=0.05, overlap=0.5,
    version="1", api_key="N5Xs9o02pFDsaVc5pjcd", rgb=True,
    depth=False, device=None, blocking=True)
    # Running our model and displaying the video output with detections
    while True:
        t0 = time.time()
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect()
        predictions = result["predictions"]
        #{
        #    predictions:
        #    [ {
        #        x: (middle),
        #        y:(middle),
        #        width:
        #        height:
        #        depth: ###->
        #        confidence:
        #        class:
        #        mask: {
        #    ]
        #}
        #frame - frame after preprocs, with predictions
        #raw_frame - original frame from your OAK
        #depth - depth map for raw_frame, center-rectified to the center camera

        # timing: for benchmarking purposes
        t = time.time()-t0
        print("FPS ", 1/t)
        print("PREDICTIONS ", [p.json() for p in predictions])

        # setting parameters for depth calculation
        # comment out the following 2 lines out if you're using an OAK without Depth
        #max_depth = np.amax(depth)
        #cv2.imshow("depth", depth/max_depth)
        # displaying the video feed as successive frames
        cv2.imshow("frame", frame)

        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break