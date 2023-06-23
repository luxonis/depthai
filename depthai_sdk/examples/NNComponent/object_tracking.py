from depthai_sdk import OakCamera
import depthai as dai

with OakCamera() as oak:
    color = oak.create_camera('color')
    # List of models that are supported out-of-the-box by the SDK:
    # https://docs.luxonis.com/projects/sdk/en/latest/features/ai_models/#sdk-supported-models
    nn = oak.create_nn('yolov6nr3_coco_640x352', color, tracker=True)

    nn.config_nn(resize_mode='stretch')
    nn.config_tracker(
        tracker_type=dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
        track_labels=[0], # Track only 1st object from the object map. If unspecified, track all object types
        # track_labels=['person'] # Track only people (for coco datasets, person is 1st object in the map)
        assignment_policy=dai.TrackerIdAssignmentPolicy.SMALLEST_ID,
        max_obj=10, # Max objects to track, which can improve performance
        threshold=0.1 # Tracker threshold
    )

    oak.visualize([nn.out.tracker], fps=True)
    oak.visualize(nn.out.passthrough)
    oak.start(blocking=True)
