from depthai_sdk import OakCamera

def cb(packet):
    pass

with OakCamera() as oak:
    color = oak.create_camera('color', fps=20)
    det = oak.create_nn('yolov6nr3_coco_640x352', color) # Detect people
    # Currently, only crop is supported for 2-stage pipelines
    det.config_nn(resize_mode='crop')

    face_nn = oak.create_nn('face-detection-retail-0004', det)
    # # Only run face detection on detected people (0th label in yolov7tiny coco is 'person')
    face_nn.config_multistage_nn(labels=[0])
    oak.visualize([face_nn], fps=True, callback=cb)
    oak.visualize(det.out.passthrough)
    oak.start(blocking=True)