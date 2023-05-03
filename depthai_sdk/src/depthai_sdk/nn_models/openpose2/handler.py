import cv2
import numpy as np

from depthai_sdk import toTensorResult, Previews

keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank',
                    'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28],
          [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]


def getKeypoints(probMap, threshold=0.2):
    mapSmooth = cv2.GaussianBlur(probMap, (3, 3), 0, 0)
    mapMask = np.uint8(mapSmooth > threshold)
    keypoints = []
    contours = None
    try:
        # OpenCV4.x
        contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # OpenCV3.x
        _, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints


def getValidPairs(outputs, w, h, detectedKeypoints):
    validPairs = []
    invalidPairs = []
    nInterpSamples = 10
    pafScoreTh = 0.2
    confTh = 0.4
    for k in range(len(mapIdx)):

        pafA = outputs[0, mapIdx[k][0], :, :]
        pafB = outputs[0, mapIdx[k][1], :, :]
        pafA = cv2.resize(pafA, (w, h))
        pafB = cv2.resize(pafB, (w, h))
        candA = detectedKeypoints[POSE_PAIRS[k][0]]

        candB = detectedKeypoints[POSE_PAIRS[k][1]]

        nA = len(candA)
        nB = len(candB)

        if (nA != 0 and nB != 0):
            validPair = np.zeros((0, 3))
            for i in range(nA):
                maxJ = -1
                maxScore = -1
                found = 0
                for j in range(nB):
                    d_ij = np.subtract(candB[j][:2], candA[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=nInterpSamples),
                                            np.linspace(candA[i][1], candB[j][1], num=nInterpSamples)))
                    pafInterp = []
                    for k in range(len(interp_coord)):
                        pafInterp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    pafScores = np.dot(pafInterp, d_ij)
                    avgPafScore = sum(pafScores) / len(pafScores)

                    if (len(np.where(pafScores > pafScoreTh)[0]) / nInterpSamples) > confTh:
                        if avgPafScore > maxScore:
                            maxJ = j
                            maxScore = avgPafScore
                            found = 1
                if found:
                    validPair = np.append(validPair, [[candA[i][3], candB[maxJ][3], maxScore]], axis=0)

            validPairs.append(validPair)
        else:
            invalidPairs.append(k)
            validPairs.append([])
    return validPairs, invalidPairs


def getPersonwiseKeypoints(validPairs, invalidPairs, keypointsList):
    personwiseKeypoints = -1 * np.ones((0, 19))

    for k in range(len(mapIdx)):
        if k not in invalidPairs:
            partAs = validPairs[k][:, 0]
            partBs = validPairs[k][:, 1]
            indexA, indexB = np.array(POSE_PAIRS[k])

            for i in range(len(validPairs[k])):
                found = 0
                personIdx = -1
                for j in range(len(personwiseKeypoints)):
                    if personwiseKeypoints[j][indexA] == partAs[i]:
                        personIdx = j
                        found = 1
                        break

                if found:
                    personwiseKeypoints[personIdx][indexB] = partBs[i]
                    personwiseKeypoints[personIdx][-1] += keypointsList[partBs[i].astype(int), 2] + validPairs[k][i][
                        2]

                elif not found and k < 17:
                    row = -1 * np.ones(19)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = sum(keypointsList[validPairs[k][i, :2].astype(int), 2]) + validPairs[k][i][2]
                    personwiseKeypoints = np.vstack([personwiseKeypoints, row])
    return personwiseKeypoints


threshold = 0.3
nPoints = 18
detectedKeypoints = []


def decode(nnManager, packet):
    outputs = toTensorResult(packet)["Openpose/concat_stage7"].astype('float32')
    w, h = nnManager.inputSize

    detectedKeypoints = []
    keypointsList = np.zeros((0, 3))
    keypointId = 0

    for part in range(nPoints):
        probMap = outputs[0, part, :, :]
        probMap = cv2.resize(probMap, (w, h))  # (456, 256)
        keypoints = getKeypoints(probMap, threshold)
        keypointsWithId = []

        for i in range(len(keypoints)):
            keypointsWithId.append(keypoints[i] + (keypointId,))
            keypointsList = np.vstack([keypointsList, keypoints[i]])
            keypointId += 1

        detectedKeypoints.append(keypointsWithId)

    validPairs, invalidPairs = getValidPairs(outputs, w, h, detectedKeypoints)
    personwiseKeypoints = getPersonwiseKeypoints(validPairs, invalidPairs, keypointsList)
    keypointsLimbs = [detectedKeypoints, personwiseKeypoints, keypointsList]

    return keypointsLimbs


def draw(nnManager, keypointsLimbs, frames):
    for name, frame in frames:
        if name == "color" and nnManager.source == "color" and not nnManager._fullFov:
            scaleFactor = frame.shape[0] / nnManager.inputSize[1]
            offsetW = int(frame.shape[1] - nnManager.inputSize[0] * scaleFactor) // 2

            def scale(point):
                return int(point[0] * scaleFactor) + offsetW, int(point[1] * scaleFactor)
        elif name in (Previews.color.name, Previews.nnInput.name, "host"):
            scaleH = frame.shape[0] / nnManager.inputSize[1]
            scaleW = frame.shape[1] / nnManager.inputSize[0]

            def scale(point):
                return int(point[0] * scaleW), int(point[1] * scaleH)
        else:
            continue

        if len(keypointsLimbs) == 3:
            detectedKeypoints = keypointsLimbs[0]
            personwiseKeypoints = keypointsLimbs[1]
            keypointsList = keypointsLimbs[2]

            for i in range(nPoints):
                for j in range(len(detectedKeypoints[i])):
                    cv2.circle(frame, scale(detectedKeypoints[i][j][0:2]), 5, colors[i], -1, cv2.LINE_AA)
            for i in range(17):
                for n in range(len(personwiseKeypoints)):
                    index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                    if -1 in index:
                        continue
                    B = np.int32(keypointsList[index.astype(int), 0])
                    A = np.int32(keypointsList[index.astype(int), 1])
                    cv2.line(frame, scale((B[0], A[0])), scale((B[1], A[1])), colors[i], 3, cv2.LINE_AA)

