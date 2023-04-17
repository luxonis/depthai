from depthai_sdk import OakCamera, RecordType

with OakCamera() as camera:
    rgb = camera.create_camera('color', encode="h265", fps=30)
    right = camera.create_camera('right')
    nn = camera.create_nn('mobilenet-ssd', right)
    camera.record([rgb], 'test', record_type=RecordType.VIDEO)
    camera.visualize([nn], fps=True)
    camera.start() # Start the pipeline (upload it to the OAK)

    while camera.running():
        
        # Since we are not in blocking mode, we have to poll oak camera to
        # visualize frames, call callbacks, process keyboard keys, etc.
        #time.sleep(1)
        camera.poll()