from depthai_sdk import OakCamera

with OakCamera(recording='./1-18443010D116631200') as oak:
    # Create stereo component, initialize left/right MonoCamera nodes for 800P and 60FPS
    stereo = oak.create_stereo('800p', fps=60)
    # Visualize normalized and colorized disparity stream
    oak.visualize(stereo.out.disparity)
    # Start the pipeline, continuously poll
    oak.show_graph()
    oak.start(blocking=True)
