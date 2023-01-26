from depthai_sdk import OakCamera, RecordType

with OakCamera(replay='record/lr') as oak:
    color = oak.create_camera('color,c', resolution='1200p', fps=20)
    color.config_camera(size=(1200, 800))
    oak.replay.disableStream('color,c')
    left = oak.create_camera('left,c', resolution='1200p', fps=20)
    left.config_camera(size=(1200, 800))
    right = oak.create_camera('right,c', resolution='1200p', fps=20)
    right.config_camera(size=(1200, 800))

    stereo = oak.create_stereo(left=left, right=right)

    # Sync & save all (encoded) streams
    # oak.record([color.out.encoded, left.out.encoded, right.out.encoded], './record', RecordType.VIDEO)
    oak.visualize([color, stereo], fps=True)

    oak.show_graph()

    oak.start(blocking=True)
