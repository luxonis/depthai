from depthai_sdk import OakCamera

with OakCamera() as oak:
    color = oak.create_camera('color')
    face_det = oak.create_nn('face-detection-retail-0004', color)
    # Control the camera's exposure/focus based on the (largest) detected face
    color.control_with_nn(face_det, auto_focus=True, auto_exposure=True, debug=False)

    oak.visualize(face_det, fps=True)
    oak.start(blocking=True)
