from depthai_sdk import OakCamera, RecordType

with OakCamera() as oak:
    color = oak.create_camera('color', resolution='1080P', fps=10, encode='H265')
    left = oak.create_camera('left', resolution='800p', fps=10, encode='H265')
    right = oak.create_camera('right', resolution='800p', fps=10, encode='H265')

    stereo = oak.create_stereo(left=left, right=right)
    nn = oak.create_nn('mobilenet-ssd', color, spatial=stereo)

    # Sync & save all (encoded) streams
    oak.record([color.out.encoded, left.out.encoded, right.out.encoded], './record', RecordType.VIDEO) \
        .configure_syncing(enable_sync=True, threshold_ms=50)

    oak.visualize([color.out.encoded], fps=True)

    oak.start(blocking=True)
