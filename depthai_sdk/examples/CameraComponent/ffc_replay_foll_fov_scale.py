from depthai_sdk import OakCamera, ResizeMode, Visualizer, FramePacket
import cv2
import depthai as dai
import numpy as np


def getMesh(calibData, resolution, offset, rectificationScale):
    print("------mesh res", resolution, "offset", offset) # TODO see if offset is needed here and implement...
    width, height = resolution
    offsetWidth, offsetHeight = offset
    ## Top left and bottom right are from camera perspective where Top left corner is at (0,0) and bottom right is at (width, height)
    topLeftPixel = dai.Point2f(offsetWidth, offsetHeight) 
    bottomRightPixel = dai.Point2f(resolution[0] + offsetWidth , resolution[1] + offsetHeight)

    print(topLeftPixel.x, topLeftPixel.y)
    print(bottomRightPixel.x, bottomRightPixel.y)
    def_m1, w, h = calibData.getDefaultIntrinsics(calibData.getStereoLeftCameraId())
    """ def_m1 = np.array(def_m1) * 1.5
    def_m1[0,2] -= (1920-1280) / 2 """


    print("Def Intrins---")
    print(w,h)
    print(np.array(def_m1))
    # we are using original height and width since we don't want to resize. But just crop based on the corner pixel positions
    M1 = np.array(calibData.getCameraIntrinsics(calibData.getStereoLeftCameraId(), w, h, topLeftPixel, bottomRightPixel))
    M2 = np.array(calibData.getCameraIntrinsics(calibData.getStereoRightCameraId(), w, h, topLeftPixel, bottomRightPixel))
    print("cropped Intrins")
    print(resolution)
    print(M1)
    
    d1 = np.array(calibData.getDistortionCoefficients(calibData.getStereoLeftCameraId()))
    d2 = np.array(calibData.getDistortionCoefficients(calibData.getStereoRightCameraId()))

    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    R2 = np.array(calibData.getStereoRightRectificationRotation())

    tranformation = np.array(calibData.getCameraExtrinsics(calibData.getStereoLeftCameraId(), calibData.getStereoRightCameraId()))
    R = tranformation[:3, :3]
    T = tranformation[:3, 3]

    debug = False
    if debug: 
        print('printing transformation matrix')
        print(tranformation)

        print(R)
        print('printing Tranlsation vec')

        print(T)
        print('Printing old R1 and R2')
        print(R1)
        print(R2)

    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    #                                                             M1,
    #                                                             d1,
    #                                                             M2,
    #                                                             d2,
    #                                                             resolution, R, T)

    rectIntrinsics = M2.copy()

    if rectificationScale > 0 and rectificationScale < 1:
        rectIntrinsics[0][0] *= rectificationScale
        rectIntrinsics[1][1] *= rectificationScale

    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, rectIntrinsics, resolution, cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, rectIntrinsics, resolution, cv2.CV_32FC1)

    meshCellSize = 16
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    meshLeft = list(meshLeft.tobytes())
    meshRight = list(meshRight.tobytes())
    return meshLeft, meshRight, rectIntrinsics

path = '1m'
with OakCamera(replay=path) as oak:
    oak.replay.setFps(3)
    originalRes = (1920, 1200)
    res = (1280, 720)
    scale = 0.7
    # res = (1280, 720)
    # resizeMode = ResizeMode.FULL_CROP
    resizeMode = ResizeMode.LETTERBOX
    
    cama = oak.create_camera('cama,c', resolution='1200p') # 1920x1200
    cama.config_camera(size=res)

    camb = oak.create_camera('camb,c', resolution='1200p')
    camb.config_camera(size=res, resize_mode=resizeMode)
    # camb.config_camera(size=res)

    camc = oak.create_camera('camc,c', resolution='1200p')
    camc.config_camera(size=res, resize_mode=resizeMode)
    # camb.config_camera(size=res)

    stereo = oak.create_stereo(left=camb, right=camc)
    # stereo.config_undistortion(M2_offset=0)

    oak.visualize([stereo.out.disparity, stereo.out.rectified_right, camc], fps=True)
    pipeline = oak.build()

    # calibData = device.readCalibration()
    calibData = oak.replay._calibData
    leftMesh, rightMesh, rectIntrinsics = getMesh(calibData, res, (0, 0), scale)
    stereo.node.loadMeshData(leftMesh, rightMesh)

    def disp(packet: FramePacket, vis: Visualizer):
        dispFrame = packet.imgFrame.getFrame()
        cv2.imshow('disp', dispFrame)

    oak.callback(stereo.out.disparity, disp)

    oak.start(blocking=True)
