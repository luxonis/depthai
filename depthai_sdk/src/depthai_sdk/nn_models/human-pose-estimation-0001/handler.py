import cv2
import numpy as np
from depthai import NNData

from depthai_sdk.classes.nn_results import ImgLandmarks

KEYPOINTS_MAPPING = [
    'Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr',
    'L-Sho', 'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee',
    'R-Ank', 'L-Hip', 'L-Knee', 'L-Ank', 'R-Eye',
    'L-Eye', 'R-Ear', 'L-Ear'
]
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]
]
SORTED_POSE_PAIRS = list(sorted(POSE_PAIRS, key=lambda x: tuple(x)))

MAP_IDX = [
    [31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], [23, 24], [25, 26],
    [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]
]
ALL_COLORS = [
    [0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
    [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
    [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]
]

NN_HEIGHT, NN_WIDTH = 256, 456

THRESHOLD = 0.3
N_POINTS = 18


def decode(nn_data: NNData) -> ImgLandmarks:
    heatmaps = np.array(nn_data.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
    pafs = np.array(nn_data.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
    heatmaps = heatmaps.astype('float32')
    pafs = pafs.astype('float32')
    outputs = np.concatenate((heatmaps, pafs), axis=1)

    new_keypoints = []
    new_keypoints_list = np.zeros((0, 3))
    keypoint_id = 0

    for row in range(N_POINTS):
        prob_map = outputs[0, row, :, :]
        prob_map = cv2.resize(prob_map, (NN_WIDTH, NN_HEIGHT))  # (456, 256)
        keypoints = get_keypoints(prob_map, threshold=THRESHOLD)
        new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
        keypoints_with_id = []

        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoint_id += 1

        new_keypoints.append(keypoints_with_id)

    valid_pairs, invalid_pairs = get_valid_pairs(outputs, w=NN_WIDTH, h=NN_HEIGHT, detected_keypoints=new_keypoints)
    new_personwise_keypoints = get_personwise_keypoints(valid_pairs, invalid_pairs, new_keypoints_list)

    keypoint_points = []
    keypoints_indices = []

    for n in range(len(new_personwise_keypoints)):
        person_keypoints = []
        indices = []
        for i in range(N_POINTS - 1):
            index = new_personwise_keypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue

            k1 = np.int32(new_keypoints_list[index.astype(int), 0]) / NN_WIDTH
            k2 = np.int32(new_keypoints_list[index.astype(int), 1]) / NN_HEIGHT
            person_keypoints.append([[k1[0], k2[0]], [k1[1], k2[1]]])
            indices.append(i)

        keypoint_points.append(person_keypoints)
        keypoints_indices.append(indices)

    return ImgLandmarks(nn_data=nn_data,
                        landmarks=keypoint_points,
                        landmarks_indices=keypoints_indices,
                        pairs=POSE_PAIRS,
                        colors=ALL_COLORS)


def get_keypoints(prob_map, threshold=0.2):
    map_smooth = cv2.GaussianBlur(prob_map, (3, 3), 0, 0)
    map_mask = np.uint8(map_smooth > threshold)
    keypoints = []

    try:
        # OpenCV4.x
        contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        # OpenCV3.x
        _, contours, _ = cv2.findContours(map_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        blob_mask = np.zeros(map_mask.shape)
        blob_mask = cv2.fillConvexPoly(blob_mask, cnt, 1)
        masked_prob_map = map_smooth * blob_mask
        _, max_val, _, max_loc = cv2.minMaxLoc(masked_prob_map)
        keypoints.append(max_loc + (prob_map[max_loc[1], max_loc[0]],))

    return keypoints


def get_valid_pairs(outputs, w, h, detected_keypoints):
    valid_pairs = []
    invalid_pairs = []
    n_interp_samples = 10
    paf_score_th = 0.2
    conf_th = 0.4

    for k in range(len(MAP_IDX)):
        paf_a = outputs[0, MAP_IDX[k][0], :, :]
        paf_b = outputs[0, MAP_IDX[k][1], :, :]
        paf_a = cv2.resize(paf_a, (w, h))
        paf_b = cv2.resize(paf_b, (w, h))

        cand_a = detected_keypoints[POSE_PAIRS[k][0]]
        cand_b = detected_keypoints[POSE_PAIRS[k][1]]

        n_a = len(cand_a)
        n_b = len(cand_b)

        if (n_a != 0 and n_b != 0):
            valid_pair = np.zeros((0, 3))
            for i in range(n_a):
                max_j = -1
                max_score = -1
                found = 0
                for j in range(n_b):
                    d_ij = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = np.linalg.norm(d_ij)
                    if norm:
                        d_ij = d_ij / norm
                    else:
                        continue
                    interp_coord = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=n_interp_samples),
                                            np.linspace(cand_a[i][1], cand_b[j][1], num=n_interp_samples)))
                    paf_interp = []
                    for k in range(len(interp_coord)):
                        paf_interp.append([paf_a[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
                                           paf_b[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))]])
                    paf_scores = np.dot(paf_interp, d_ij)
                    avg_paf_score = sum(paf_scores) / len(paf_scores)

                    if (len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples) > conf_th:
                        if avg_paf_score > max_score:
                            max_j = j
                            max_score = avg_paf_score
                            found = 1
                if found:
                    valid_pair = np.append(valid_pair, [[cand_a[i][3], cand_b[max_j][3], max_score]], axis=0)

            valid_pairs.append(valid_pair)
        else:
            invalid_pairs.append(k)
            valid_pairs.append([])

    return valid_pairs, invalid_pairs


def get_personwise_keypoints(valid_pairs, invalid_pairs, keypoints_list):
    personwise_keypoints = -1 * np.ones((0, 19))

    for k in range(len(MAP_IDX)):
        if k in invalid_pairs:
            continue

        part_as = valid_pairs[k][:, 0]
        part_bs = valid_pairs[k][:, 1]
        index_a, index_b = np.array(POSE_PAIRS[k])

        for i in range(len(valid_pairs[k])):
            found = 0
            person_idx = -1
            for j in range(len(personwise_keypoints)):
                if personwise_keypoints[j][index_a] == part_as[i]:
                    person_idx = j
                    found = 1
                    break

            if found:
                personwise_keypoints[person_idx][index_b] = part_bs[i]
                personwise_keypoints[person_idx][-1] += keypoints_list[part_bs[i].astype(int), 2] + \
                                                        valid_pairs[k][i][2]

            elif not found and k < 17:
                row = -1 * np.ones(19)
                row[index_a] = part_as[i]
                row[index_b] = part_bs[i]
                row[-1] = sum(keypoints_list[valid_pairs[k][i, :2].astype(int), 2]) + valid_pairs[k][i][2]
                personwise_keypoints = np.vstack([personwise_keypoints, row])

    return personwise_keypoints
