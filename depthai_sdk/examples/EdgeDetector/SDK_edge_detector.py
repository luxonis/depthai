from depthai_sdk import OakCamera

with OakCamera() as camera:
    rgb = camera.create_camera('color')
    edge = camera.create_edge_detector()
    camera.visualize([rgb, edge])

    # NO EGDE DETECTOR NODE






    camera.start() # Start the pipeline (upload it to the OAK)




    while camera.running():
        camera.poll()