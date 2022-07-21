from depthai_sdk import Camera
import blobconverter
import cv2

cam = Camera()
color = cam.create_camera('color', out='color')
cam.create_nn(blobconverter.from_zoo('mobilenet-ssd', shaves=6), color, out='dets', type='mobilenet')

cam.start()

while cam.running():
    msgs = cam.get_msgs()
    print('new msgs', msgs['color'])
    cv2.imshow('color', msgs['color'].getCvFrame())
    if cv2.waitKey(1) == ord('q'):
        break

del cam