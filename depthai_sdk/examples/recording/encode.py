from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=10)
    left = oak.create_camera('left', resolution='800p', fps=10)
    right = oak.create_camera('right', resolution='800p', fps=10)

    color_encoder = oak.create_encoder(color, codec='h265')
    left_encoder = oak.create_encoder(left, codec='h265')
    right_encoder = oak.create_encoder(right, codec='h265')

    stereo = oak.create_stereo(left=left, right=right)
    nn = oak.create_nn('mobilenet-ssd', color, spatial=stereo)

    # Sync & save all (encoded) streams
    oak.record([color_encoder, left_encoder, right_encoder], './record', RecordType.VIDEO) \
        .configure_syncing(enable_sync=True, threshold_ms=50)

    oak.visualize([color_encoder], fps=True)

    oak.start(blocking=True)
