import numpy as np

from depthai_sdk.classes import ImgLandmarks

KEYPOINTS_MAPPING = [
    'WRIST',
    'THUMB_CMC',
    'THUMB_MCP',
    'THUMB_IP',
    'THUMB_TIP',
    'INDEX_FINGER_MCP',
    'INDEX_FINGER_PIP',
    'INDEX_FINGER_DIP',
    'INDEX_FINGER_TIP',
    'MIDDLE_FINGER_MCP',
    'MIDDLE_FINGER_PIP',
    'MIDDLE_FINGER_DIP',
    'MIDDLE_FINGER_TIP',
    'RING_FINGER_MCP',
    'RING_FINGER_PIP',
    'RING_FINGER_DIP',
    'RING_FINGER_TIP',
    'PINKY_MCP',
    'PINKY_PIP',
    'PINKY_DIP',
    'PINKY_TIP'
]

JOINT_PAIRS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20]
]

SORTED_POSE_PAIRS = list(sorted(JOINT_PAIRS, key=lambda x: tuple(x)))

ALL_COLORS = [
    [0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
    [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
    [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]
]

NN_HEIGHT, NN_WIDTH = 256, 456

THRESHOLD = 0.3
N_POINTS = 18


def decode(nn_data):
    lm_score = nn_data.getLayerFp16("Identity_1")[0]
    print(lm_score)
    # print('lm_score', lm_score)
    if lm_score > 0.5:
        handedness = nn_data.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(nn_data.getLayerFp16("Identity_dense/BiasAdd/Add")).reshape(-1, 3)
        # hand.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
        # print(lm_raw)
        norm_landmarks = lm_raw / 224.0
        # hand.norm_landmarks[:,2] /= 0.4

        landmarks = np.squeeze(norm_landmarks)[..., :2]
        print(landmarks)
        return ImgLandmarks(nn_data=nn_data,
                            landmarks=landmarks,
                            landmarks_indices=list(range(len(landmarks))),
                            pairs=JOINT_PAIRS,
                            colors=ALL_COLORS)
    else:
        return None
