import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, exp, pi, floor, sin, cos, atan2, gcd

# To not display: RuntimeWarning: overflow encountered in exp
# in line:  scores = 1 / (1 + np.exp(-scores))
np.seterr(over='ignore')

class HandRegion:
    """
        Attributes:
        pd_score : detection score
        pd_box : detection box [x, y, w, h], normalized [0,1] in the squared image
        pd_kps : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        rect_x_center, rect_y_center : center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
        rotation : rotation angle of rotated bounding rectangle with y-axis in radian
        rect_x_center_a, rect_y_center_a : center coordinates of the rotated bounding rectangle, in pixels in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, in pixels in the squared image
        rect_points : list of the 4 points coordinates of the rotated bounding rectangle, in pixels
                expressed in the squared image during processing,
                expressed in the source rectangular image when returned to the user
        lm_score: global landmark score
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks : 2D landmark coordinates in pixel in the source rectangular image
        world_landmarks : 3D landmark coordinates in meter
        handedness: float between 0. and 1., > 0.5 for right hand, < 0.5 for left hand,
        label: "left" or "right", handedness translated in a string,
        xyz: real 3D world coordinates of the wrist landmark, or of the palm center (if landmarks are not used),
        xyz_zone: (left, top, right, bottom), pixel coordinates in the source rectangular image
                of the rectangular zone used to estimate the depth
        gesture: (optional, set in recognize_gesture() when use_gesture==True) string corresponding to recognized gesture ("ONE","TWO","THREE","FOUR","FIVE","FIST","OK","PEACE")
                or None if no gesture has been recognized
        """
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def get_rotated_world_landmarks(self):
        world_landmarks_rotated = self.world_landmarks.copy()
        sin_rot = sin(self.rotation)
        cos_rot = cos(self.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        world_landmarks_rotated[:,:2] = np.dot(world_landmarks_rotated[:,:2], rot_m)
        return world_landmarks_rotated

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

class HandednessAverage:
    """
    Used to store the average handeness
    Why ? Handedness inferred by the landmark model is not perfect. For certain poses, it is not rare that the model thinks
    that a right hand is a left hand (or vice versa). Instead of using the last inferred handedness, we prefer to use the average
    of the inferred handedness on the last frames. This gives more robustness.
    """
    def __init__(self):
        self._total_handedness = 0
        self._nb = 0
    def update(self, new_handedness):
        self._total_handedness += new_handedness
        self._nb += 1
        return self._total_handedness / self._nb
    def reset(self):
        self._total_handedness = self._nb = 0


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1

        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer
    return np.array(anchors)

def generate_handtracker_anchors(input_size_width, input_size_height):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
    anchor_options = SSDAnchorOptions(num_layers=4,
                            min_scale=0.1484375,
                            max_scale=0.75,
                            input_size_height=input_size_height,
                            input_size_width=input_size_width,
                            anchor_offset_x=0.5,
                            anchor_offset_y=0.5,
                            strides=[8, 16, 16, 16],
                            aspect_ratios= [1.0],
                            reduce_boxes_in_lowest_layer=False,
                            interpolated_scale_aspect_ratio=1.0,
                            fixed_anchor_size=True)
    return generate_anchors(anchor_options)

def decode_bboxes(score_thresh, scores, bboxes, anchors, scale=128, best_only=False):
    """
    wi, hi : NN input shape
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    # Decodes the detection tensors generated by the model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.

    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 896
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 128.0
            y_scale: 128.0
            h_scale: 128.0
            w_scale: 128.0
            min_score_thresh: 0.5
            }
        }
    }
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 2016
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 192.0
            y_scale: 192.0
            w_scale: 192.0
            h_scale: 192.0
            min_score_thresh: 0.5
            }
        }
    }

    scores: shape = [number of anchors 896 or 2016]
    bboxes: shape = [ number of anchors x 18], 18 = 4 (bounding box : (cx,cy,w,h) + 14 (7 palm keypoints)
    """
    regions = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return regions
        det_scores = scores[best_id:best_id+1]
        det_bboxes2 = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return regions
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]

    # scale = 128 # x_scale, y_scale, w_scale, h_scale
    # scale = 192 # x_scale, y_scale, w_scale, h_scale

    # cx, cy, w, h = bboxes[i,:4]
    # cx = cx * anchor.w / wi + anchor.x_center
    # cy = cy * anchor.h / hi + anchor.y_center
    # lx = lx * anchor.w / wi + anchor.x_center
    # ly = ly * anchor.h / hi + anchor.y_center
    det_bboxes = det_bboxes2* np.tile(det_anchors[:,2:4], 9) / scale + np.tile(det_anchors[:,0:2],9)
    # w = w * anchor.w / wi (in the prvious line, we add anchor.x_center and anchor.y_center to w and h, we need to substract them now)
    # h = h * anchor.h / hi
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        # Decoded detection boxes could have negative values for width/height due
        # to model prediction. Filter out those boxes
        if box[2] < 0 or box[3] < 0: continue
        kps = []
        # 0 : wrist
        # 1 : index finger joint
        # 2 : middle finger joint
        # 3 : ring finger joint
        # 4 : little finger joint
        # 5 :
        # 6 : thumb joint
        # for j, name in enumerate(["0", "1", "2", "3", "4", "5", "6"]):
        #     kps[name] = det_bboxes[i,4+j*2:6+j*2]
        for kp in range(7):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        regions.append(HandRegion(float(score), box, kps))
    return regions

# Starting from opencv 4.5.4, cv2.dnn.NMSBoxes output format changed
import re
cv2_version = cv2.__version__.split('.')
v0 = int(cv2_version[0])
v1 = int(cv2_version[1])
v2 = int(re.sub(r'\D+', '', cv2_version[2]))
if  v0 > 4 or (v0 == 4 and (v1 > 5 or (v1 == 5 and v2 >= 4))):
    def non_max_suppression(regions, nms_thresh):
        # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
        # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
        # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
        # boxes = [r.box for r in regions]
        boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]
        scores = [r.pd_score for r in regions]
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) # Not using top_k=2 here because it does not give expected result. Bug ?
        return [regions[i] for i in indices]
else:
    def non_max_suppression(regions, nms_thresh):
        # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
        # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
        # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
        # boxes = [r.box for r in regions]
        boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]
        scores = [r.pd_score for r in regions]
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) # Not using top_k=2 here because it does not give expected result. Bug ?
        return [regions[i[0]] for i in indices]

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]

def detections_to_rect(regions):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # # Converts results of palm detection into a rectangle (normalized by image size)
    # # that encloses the palm and is rotated such that the line connecting center of
    # # the wrist and MCP of the middle finger is aligned with the Y-axis of the
    # # rectangle.
    # node {
    #   calculator: "DetectionsToRectsCalculator"
    #   input_stream: "DETECTION:detection"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "NORM_RECT:raw_roi"
    #   options: {
    #     [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #       rotation_vector_start_keypoint_index: 0  # Center of wrist.
    #       rotation_vector_end_keypoint_index: 2  # MCP of middle finger.
    #       rotation_vector_target_angle_degrees: 90
    #     }
    #   }

    target_angle = pi * 0.5 # 90 = pi/2
    for region in regions:

        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0] # wrist center
        x1, y1 = region.pd_kps[2] # middle finger
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)

def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]

def rect_transformation(regions, w, h):
    """
    w, h : image input shape
    """
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # # Expands and shifts the rectangle that contains the palm so that it's likely
    # # to cover the entire hand.
    # node {
    # calculator: "RectTransformationCalculator"
    # input_stream: "NORM_RECT:raw_roi"
    # input_stream: "IMAGE_SIZE:image_size"
    # output_stream: "roi"
    # options: {
    #     [mediapipe.RectTransformationCalculatorOptions.ext] {
    #     scale_x: 2.6
    #     scale_y: 2.6
    #     shift_y: -0.5
    #     square_long: true
    #     }
    # }
    # IMHO 2.9 is better than 2.6. With 2.6, it may happen that finger tips stay outside of the bouding rotated rectangle
    scale_x = 2.9
    scale_y = 2.9
    shift_x = 0
    shift_y = -0.5
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation)) #/ w
            y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation)) #/ h
            region.rect_x_center_a = region.rect_x_center*w + x_shift
            region.rect_y_center_a = region.rect_y_center*h + y_shift

        # square_long: true
        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation)

def hand_landmarks_to_rect(hand):
    # Calculates the ROI for the next frame from the current hand landmarks
    id_wrist = 0
    id_index_mcp = 5
    id_middle_mcp = 9
    id_ring_mcp =13

    lms_xy =  hand.landmarks[:,:2]
    # print(lms_xy)
    # Compute rotation
    x0, y0 = lms_xy[id_wrist]
    x1, y1 = 0.25 * (lms_xy[id_index_mcp] + lms_xy[id_ring_mcp]) + 0.5 * lms_xy[id_middle_mcp]
    rotation = 0.5 * pi - atan2(y0 - y1, x1 - x0)
    rotation = normalize_radians(rotation)
    # Now we work only on a subset of the landmarks
    ids_for_bounding_box = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    lms_xy = lms_xy[ids_for_bounding_box]
    # Find center of the boundaries of landmarks
    axis_aligned_center = 0.5 * (np.min(lms_xy, axis=0) + np.max(lms_xy, axis=0))
    # Find boundaries of rotated landmarks
    original = lms_xy - axis_aligned_center
    c, s = np.cos(rotation), np.sin(rotation)
    rot_mat = np.array(((c, -s), (s, c)))
    projected = original.dot(rot_mat)
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    projected_center = 0.5 * (min_proj + max_proj)
    center = rot_mat.dot(projected_center) + axis_aligned_center
    width, height = max_proj - min_proj
    next_hand = HandRegion()
    next_hand.rect_w_a = next_hand.rect_h_a = 2 * max(width, height)
    next_hand.rect_x_center_a = center[0] + 0.1 * height * s
    next_hand.rect_y_center_a = center[1] - 0.1 * height * c
    next_hand.rotation = rotation
    next_hand.rect_points = rotated_rect_to_points(next_hand.rect_x_center_a, next_hand.rect_y_center_a, next_hand.rect_w_a, next_hand.rect_h_a, next_hand.rotation)
    return next_hand

def warp_rect_img(rect_points, img, w, h):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (h, 0), (h, w)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (w, h))

def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return np.linalg.norm(a-b)

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    resolution: sensor resolution (width, height)
    is_height : boolean that indicates if the value 'size' represents the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288 (first compatible size > lm_input_size)
    if size < 288:
        size = 288

    width, height = resolution

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = height
        other = width
    else:
        reference = width
        other = height
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)

    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]

def recognize_gesture(hand):
    # Finger states
    # state: -1=unknown, 0=close, 1=open
    d_3_5 = distance(hand.norm_landmarks[3], hand.norm_landmarks[5])
    d_2_3 = distance(hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle0 = angle(hand.norm_landmarks[0], hand.norm_landmarks[1], hand.norm_landmarks[2])
    angle1 = angle(hand.norm_landmarks[1], hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle2 = angle(hand.norm_landmarks[2], hand.norm_landmarks[3], hand.norm_landmarks[4])
    hand.thumb_angle = angle0+angle1+angle2
    if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2:
        hand.thumb_state = 1
    else:
        hand.thumb_state = 0

    if hand.norm_landmarks[8][1] < hand.norm_landmarks[7][1] < hand.norm_landmarks[6][1]:
        hand.index_state = 1
    elif hand.norm_landmarks[6][1] < hand.norm_landmarks[8][1]:
        hand.index_state = 0
    else:
        hand.index_state = -1

    if hand.norm_landmarks[12][1] < hand.norm_landmarks[11][1] < hand.norm_landmarks[10][1]:
        hand.middle_state = 1
    elif hand.norm_landmarks[10][1] < hand.norm_landmarks[12][1]:
        hand.middle_state = 0
    else:
        hand.middle_state = -1

    if hand.norm_landmarks[16][1] < hand.norm_landmarks[15][1] < hand.norm_landmarks[14][1]:
        hand.ring_state = 1
    elif hand.norm_landmarks[14][1] < hand.norm_landmarks[16][1]:
        hand.ring_state = 0
    else:
        hand.ring_state = -1

    if hand.norm_landmarks[20][1] < hand.norm_landmarks[19][1] < hand.norm_landmarks[18][1]:
        hand.little_state = 1
    elif hand.norm_landmarks[18][1] < hand.norm_landmarks[20][1]:
        hand.little_state = 0
    else:
        hand.little_state = -1

    # Gesture
    if hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 1 and hand.little_state == 1:
        hand.gesture = "FIVE"
    elif hand.thumb_state == 0 and hand.index_state == 0 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "FIST"
    elif hand.thumb_state == 1 and hand.index_state == 0 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "OK"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "PEACE"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "ONE"
    elif hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "TWO"
    elif hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "THREE"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 1 and hand.little_state == 1:
        hand.gesture = "FOUR"
    else:
        hand.gesture = None


# Movenet

class Body:
    def __init__(self, scores=None, keypoints_norm=None, keypoints=None, score_thresh=None, crop_region=None, next_crop_region=None):
        """
        Attributes:
        scores : scores of the keypoints
        keypoints_norm : keypoints normalized ([0,1]) coordinates (x,y) in the squared cropped region
        keypoints_square : keypoints coordinates (x,y) in pixels in the square padded image
        keypoints : keypoints coordinates (x,y) in pixels in the source image (not padded)
        score_thresh : score threshold used
        crop_region : cropped region on which the current body was inferred
        next_crop_region : cropping region calculated from the current body keypoints and that will be used on next frame
        """
        self.scores = scores
        self.keypoints_norm = keypoints_norm
        self.keypoints = keypoints
        self.score_thresh = score_thresh
        self.crop_region = crop_region
        self.next_crop_region = next_crop_region
        # self.keypoints_square = (self.keypoints_norm * self.crop_region.size).astype(np.int32)
        self.keypoints = (np.array([self.crop_region.xmin, self.crop_region.ymin]) + self.keypoints_norm * self.crop_region.size).astype(np.int32)

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

    def distance_to_wrist(self, hand, wrist_handedness, pad_w=0, pad_h=0):
        """
        Calculate the distance between a hand (class Hand) wrist position
        and one of the body wrist given by wrist_handedness (= "left" or "right")
        As the hand.landmarks cooordinates are expressed in the padded image, we must substract the padding (given by pad_w and pad_w)
        to be coherent with the body keypoint coordinates which are expressed in the source image.
        """
        return distance(hand.landmarks[0]-np.array([pad_w, pad_h]), self.keypoints[BODY_KP[wrist_handedness+'_wrist']])

CropRegion = namedtuple('CropRegion',['xmin', 'ymin', 'xmax',  'ymax', 'size']) # All values are in pixel. The region is a square of size 'size' pixels

# Dictionary that maps from joint names to keypoint indices.
BODY_KP = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

class BodyPreFocusing:
    """
    Body Pre Focusing with Movenet
    Contains all is needed for :
    - Movenet smart cropping (determines from the body detected in frame N,
    the region of frame N+1 ow which the Movenet inference is run).
    - Body Pre Focusing (determining from the Movenet wrist keypoints a smaller zone
    on which Palm detection is run).
    Both Smart cropping and Body Pre Focusing are important for model accuracy when
    the body is far.
    """
    def __init__(self, img_w, img_h, pad_w, pad_h, frame_size, mode="group", score_thresh=0.2, scale=1.0, hands_up_only=True):

        self.img_w = img_w
        self.img_h = img_h
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.frame_size = frame_size
        self.mode = mode
        self.score_thresh = score_thresh
        self.scale = scale
        self.hands_up_only = hands_up_only
        # Defines the default crop region (pads the full image from both sides to make it a square image)
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        self.init_crop_region = CropRegion(-self.pad_w, -self.pad_h,-self.pad_w+self.frame_size, -self.pad_h+self.frame_size, self.frame_size)


    def torso_visible(self, scores):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[BODY_KP['left_hip']] > self.score_thresh or
                scores[BODY_KP['right_hip']] > self.score_thresh) and
                (scores[BODY_KP['left_shoulder']] > self.score_thresh or
                scores[BODY_KP['right_shoulder']] > self.score_thresh))

    def determine_torso_and_body_range(self, keypoints, scores, center_x, center_y):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - keypoints[BODY_KP[joint]][1])
            dist_x = abs(center_x - keypoints[BODY_KP[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(BODY_KP)):
            if scores[i] < self.score_thresh:
                continue
            dist_y = abs(center_y - keypoints[i][1])
            dist_x = abs(center_x - keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, body):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        if self.torso_visible(body.scores):
            center_x = (body.keypoints[BODY_KP['left_hip']][0] + body.keypoints[BODY_KP['right_hip']][0]) // 2
            center_y = (body.keypoints[BODY_KP['left_hip']][1] + body.keypoints[BODY_KP['right_hip']][1]) // 2
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self.determine_torso_and_body_range(body.keypoints, body.scores, center_x, center_y)
            crop_length_half = np.amax([max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x, center_y, self.img_h - center_y])
            crop_length_half = int(round(np.amin([crop_length_half, np.amax(tmp)])))
            crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return CropRegion(crop_corner[0], crop_corner[1], crop_corner[0]+crop_length, crop_corner[1]+crop_length,crop_length)
        else:
            return self.init_crop_region

    """
    Body Pre Focusing stuff
    """
    def torso_visible(self, scores):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[BODY_KP['left_hip']] > self.score_thresh or
                scores[BODY_KP['right_hip']] > self.score_thresh) and
                (scores[BODY_KP['left_shoulder']] > self.score_thresh or
                scores[BODY_KP['right_shoulder']] > self.score_thresh))

    def determine_torso_and_body_range(self, keypoints, scores, center_x, center_y):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - keypoints[BODY_KP[joint]][1])
            dist_x = abs(center_x - keypoints[BODY_KP[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(BODY_KP)):
            if scores[i] < self.score_thresh:
                continue
            dist_y = abs(center_y - keypoints[i][1])
            dist_x = abs(center_x - keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, body):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        if self.torso_visible(body.scores):
            center_x = (body.keypoints[BODY_KP['left_hip']][0] + body.keypoints[BODY_KP['right_hip']][0]) // 2
            center_y = (body.keypoints[BODY_KP['left_hip']][1] + body.keypoints[BODY_KP['right_hip']][1]) // 2
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self.determine_torso_and_body_range(body.keypoints, body.scores, center_x, center_y)
            crop_length_half = np.amax([max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x, center_y, self.img_h - center_y])
            crop_length_half = int(round(np.amin([crop_length_half, np.amax(tmp)])))
            crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return CropRegion(crop_corner[0], crop_corner[1], crop_corner[0]+crop_length, crop_corner[1]+crop_length,crop_length)
        else:
            return self.init_crop_region

    def estimate_focus_zone_size(self, body):
        """
        This function is called if at least the segment "wrist_elbow" is visible.
        We calculate the length of every segment from a predefined list. A segment length
        is the distance between the 2 endpoints weighted by a coefficient. The weight have been chosen
        so that the length of all segments are roughly equal. We take the maximal length to estimate
        the size of the focus zone.
        If no segment are vissible, we consider the body is very close
        to the camera, and therefore there is no need to focus. Return 0
        To not have at least one shoulder and one hip visible means the body is also very close
        and the estimated size needs to be adjusted (bigger)
        """
        segments = [
            ("left_shoulder", "left_elbow", 2.3),
            ("left_elbow", "left_wrist", 2.3),
            ("left_shoulder", "left_hip", 1),
            ("left_shoulder", "right_shoulder", 1.5),
            ("right_shoulder", "right_elbow", 2.3),
            ("right_elbow", "right_wrist", 2.3),
            ("right_shoulder", "right_hip", 1),
        ]
        lengths = []
        for s in segments:
            if body.scores[BODY_KP[s[0]]] > self.score_thresh and body.scores[BODY_KP[s[1]]] > self.score_thresh:
                l = np.linalg.norm(body.keypoints[BODY_KP[s[0]]] - body.keypoints[BODY_KP[s[1]]])
                lengths.append(l)
        if lengths:
            if ( body.scores[BODY_KP["left_hip"]] < self.score_thresh and
                body.scores[BODY_KP["right_hip"]] < self.score_thresh or
                body.scores[BODY_KP["left_shoulder"]] < self.score_thresh and
                body.scores[BODY_KP["right_shoulder"]] < self.score_thresh) :
                coef = 1.5
            else:
                coef = 1.0
            return 2 * int(coef * self.scale * max(lengths) / 2) # The size is made even
        else:
            return 0

    def get_focus_zone(self, body):
        """
        Return a tuple (focus_zone, label)
        'body' = instance of class Body
        'focus_zone' is a zone around a hand or hands, depending on the value
        of self.mode ("left", "right", "higher" or "group") and on the value of self.hands_up_only.
            - self.mode = "left" (resp "right"): we are looking for the zone around the left (resp right) wrist,
            - self.mode = "group": the zone encompasses both wrists,
            - self.mode = "higher": the zone is around the higher wrist (smaller y value),
            - self.hands_up_only = True: we don't take into consideration the wrist if the corresponding elbow is above the wrist,
        focus_zone is a list [left, top, right, bottom] defining the top-left and right-bottom corners of a square.
        Values are expressed in pixels in the source image C.S.
        The zone is constrained to the squared source image (= source image with padding if necessary).
        It means that values can be negative.
        left and right in [-pad_w, img_w + pad_w]
        top and bottom in [-pad_h, img_h + pad_h]
        'label' describes which wrist keypoint(s) were used to build the zone : "left", "right" or "group" (if built from both wrists)

        If the wrist keypoint(s) is(are) not present or is(are) present but self.hands_up_only = True and
        wrist(s) is(are) below corresponding elbow(s), then focus_zone = None.
        """

        def zone_from_center_size(x, y, size):
            """
            Return zone [left, top, right, bottom]
            from zone center (x,y) and zone size (the zone is square).
            """
            half_size = size // 2
            size = half_size * 2
            if size > self.img_w:
                x = self.img_w // 2
            x1 = x - half_size
            if x1 < -self.pad_w:
                x1 = -self.pad_w
            elif x1 + size > self.img_w + self.pad_w:
                x1 = self.img_w + self.pad_w - size
            x2 = x1 + size
            if size > self.img_h:
                y = self.img_h // 2
            y1 = y - half_size
            if y1 < -self.pad_h:
                y1 = -self.pad_h
            elif y1 + size > self.img_h + self.pad_h:
                y1 = self.img_h + self.pad_h - size
            y2 = y1 + size
            return [x1, y1, x2, y2]


        def get_one_hand_zone(hand_label):
            """
            Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
            Values are expressed in pixels in the source image C.S.
            If the wrist keypoint is not visible, return None.
            If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.
            """
            wrist_kp = hand_label + "_wrist"
            wrist_score = body.scores[BODY_KP[wrist_kp]]
            if wrist_score < self.score_thresh:
                return None
            x, y = body.keypoints[BODY_KP[wrist_kp]]
            if self.hands_up_only:
                # We want to detect only hands where the wrist is above the elbow (when visible)
                elbow_kp = hand_label + "_elbow"
                if body.scores[BODY_KP[elbow_kp]] > self.score_thresh and \
                    body.keypoints[BODY_KP[elbow_kp]][1] < body.keypoints[BODY_KP[wrist_kp]][1]:
                    return None
            # Let's evaluate the size of the focus zone
            size = self.estimate_focus_zone_size(body)
            if size == 0: return [-self.pad_w, -self.pad_h, self.frame_size-self.pad_w, self.frame_size-self.pad_h] # The hand is too close. No need to focus
            return zone_from_center_size(x, y, size)

        if self.mode == "group":
            zonel = get_one_hand_zone("left")
            if zonel:
                zoner = get_one_hand_zone("right")
                if zoner:
                    xl1, yl1, xl2, yl2 = zonel
                    xr1, yr1, xr2, yr2 = zoner
                    x1 = min(xl1, xr1)
                    y1 = min(yl1, yr1)
                    x2 = max(xl2, xr2)
                    y2 = max(yl2, yr2)
                    # Global zone center (x,y)
                    x = int((x1+x2)/2)
                    y = int((y1+y2)/2)
                    size_x = x2-x1
                    size_y = y2-y1
                    size = 2 * (max(size_x, size_y) // 2)
                    return (zone_from_center_size(x, y, size), "group")
                else:
                    return (zonel, "left")
            else:
                return (get_one_hand_zone("right"), "right")
        elif self.mode == "higher":
            if body.scores[BODY_KP["left_wrist"]] > self.score_thresh:
                if body.scores[BODY_KP["right_wrist"]] > self.score_thresh:
                    if body.keypoints[BODY_KP["left_wrist"]][1] > body.keypoints[BODY_KP["right_wrist"]][1]:
                        hand_label = "right"
                    else:
                        hand_label = "left"
                else:
                    hand_label = "left"
            else:
                if body.scores[BODY_KP["right_wrist"]] > self.score_thresh:
                    hand_label = "right"
                else:
                    return (None, None)
            return (get_one_hand_zone(hand_label), hand_label)
        else: # "left" or "right"
            return (get_one_hand_zone(self.mode), self.mode)


