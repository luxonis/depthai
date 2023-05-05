from depthai_sdk import OakCamera

with OakCamera(replay='https://www.youtube.com/watch?v=Y1jTEyb3wiI') as oak:
    oak.replay.set_loop(True)  # <--- Enable looping of the video, so it will never end

    color = oak.create_camera('color')
    nn = oak.create_nn('vehicle-detection-0202', color)
    oak.visualize(nn, fps=True)
    oak.start(blocking=True)
