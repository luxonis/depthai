import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, pi, floor, sin, cos, atan2, gcd
from collections import  namedtuple

# To not display: RuntimeWarning: overflow encountered in exp
# in line:  scores = 1 / (1 + np.exp(-scores))
np.seterr(over='ignore')

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

class Body:
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
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
            expressed in the original rectangular image when returned to the user
        lm_score: global landmark score
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks : 3D landmarks coordinates in the rotated bounding rectangle, in pixel in the original rectangular image
        world_landmarks : 3D landmarks coordinates in meter with mid hips point being the origin.
            The y value of landmarks_world coordinates is negative for landmarks
            above the mid hips (like shoulders) and negative for landmarks below (like feet)
        xyz: (optionally) 3D location in camera coordinate system of reference point (mid hips or mid shoulders)
        xyz_ref: (optionally) name of the reference point ("mid_hips" or "mid_shoulders"),
        xyz_zone: (optionally) 4 int array of zone (in the source image) on which is measured depth.
            xyz_zone[0:2] is top-left zone corner in pixels, xyz_zone[2:4] is bottom-right zone corner
        """
        self.pd_score = pd_score
        self.pd_box = pd_box
        self.pd_kps = pd_kps
        self.lm_raw = None
    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


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

def generate_blazepose_anchors():
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
    anchor_options = SSDAnchorOptions(
                                num_layers=5,
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=224,
                                input_size_width=224,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 32, 32, 32],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
    return generate_anchors(anchor_options)

def decode_bboxes(score_thresh, scores, bboxes, anchors, best_only=False):
    """
    wi, hi : NN input shape
    https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
    # Decodes the detection tensors generated by the TensorFlow Lite model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.
    Version 0.8.3.1:
    node {
    calculator: "TensorsToDetectionsCalculator"
    input_stream: "TENSORS:detection_tensors"
    input_side_packet: "ANCHORS:anchors"
    output_stream: "DETECTIONS:unfiltered_detections"
    options: {
        [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
        num_classes: 1
        num_boxes: 896
        num_coords: 12
        box_coord_offset: 0
        keypoint_coord_offset: 4
        num_keypoints: 4
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

    Version 0.8.4:
    [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
      num_classes: 1
      num_boxes: 2254
      num_coords: 12
      box_coord_offset: 0
      keypoint_coord_offset: 4
      num_keypoints: 4
      num_values_per_keypoint: 2
      sigmoid_score: true
      score_clipping_thresh: 100.0
      reverse_output_order: true
      x_scale: 224.0
      y_scale: 224.0
      h_scale: 224.0
      w_scale: 224.0
      min_score_thresh: 0.5
    }

    # Bounding box in each pose detection is currently set to the bounding box of
    # the detected face. However, 4 additional key points are available in each
    # detection, which are used to further calculate a (rotated) bounding box that
    # encloses the body region of interest. Among the 4 key points, the first two
    # are for identifying the full-body region, and the second two for upper body
    # only:
    #
    # Key point 0 - mid hip center
    # Key point 1 - point that encodes size & rotation (for full body)
    # Key point 2 - mid shoulder center
    # Key point 3 - point that encodes size & rotation (for upper body)
    #

    scores: shape = [number of anchors 896]
    bboxes: shape = [ number of anchors x 12], 12 = 4 (bounding box : (cx,cy,w,h) + 8 (4 palm keypoints)
    """
    bodies = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return bodies
        det_scores = scores[best_id:best_id+1]
        det_bboxes = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return bodies
        det_bboxes = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]

    scale = 224 # x_scale, y_scale, w_scale, h_scale

    # cx, cy, w, h = bboxes[i,:4]
    # cx = cx * anchor.w / wi + anchor.x_center
    # cy = cy * anchor.h / hi + anchor.y_center
    # lx = lx * anchor.w / wi + anchor.x_center
    # ly = ly * anchor.h / hi + anchor.y_center
    det_bboxes = det_bboxes* np.tile(det_anchors[:,2:4], 6) / scale + np.tile(det_anchors[:,0:2],6)
    # w = w * anchor.w / wi (in the prvious line, we add anchor.x_center and anchor.y_center to w and h, we need to substract them now)
    # h = h * anchor.h / hi
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        kps = []
        for kp in range(4):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        bodies.append(Body(float(score), box, kps))
    return bodies


def non_max_suppression(bodies, nms_thresh):

    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in bodies]
    boxes = [ [int(x*1000) for x in r.pd_box] for r in bodies]
    scores = [r.pd_score for r in bodies]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [bodies[i[0]] for i in indices]

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]

def detections_to_rect(body, kp_pair=[0,1]):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # # Converts pose detection into a rectangle based on center and scale alignment
    # # points. Pose detection contains four key points: first two for full-body pose
    # # and two more for upper-body pose.
    # node {
    #   calculator: "SwitchContainer"
    #   input_side_packet: "ENABLE:upper_body_only"
    #   input_stream: "DETECTION:detection"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "NORM_RECT:raw_roi"
    #   options {
    #     [mediapipe.SwitchContainerOptions.ext] {
    #       contained_node: {
    #         calculator: "AlignmentPointsRectsCalculator"
    #         options: {
    #           [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #             rotation_vector_start_keypoint_index: 0
    #             rotation_vector_end_keypoint_index: 1
    #             rotation_vector_target_angle_degrees: 90
    #           }
    #         }
    #       }
    #       contained_node: {
    #         calculator: "AlignmentPointsRectsCalculator"
    #         options: {
    #           [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #             rotation_vector_start_keypoint_index: 2
    #             rotation_vector_end_keypoint_index: 3
    #             rotation_vector_target_angle_degrees: 90
    #           }
    #         }
    #       }
    #     }
    #   }
    # }

    target_angle = pi * 0.5 # 90 = pi/2

    # AlignmentPointsRectsCalculator : https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
    x_center, y_center = body.pd_kps[kp_pair[0]]
    x_scale, y_scale = body.pd_kps[kp_pair[1]]
    # Bounding box size as double distance from center to scale point.
    box_size = sqrt((x_scale-x_center)**2 + (y_scale-y_center)**2) * 2
    body.rect_w = box_size
    body.rect_h = box_size
    body.rect_x_center = x_center
    body.rect_y_center = y_center

    rotation = target_angle - atan2(-(y_scale - y_center), x_scale - x_center)
    body.rotation = normalize_radians(rotation)

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

def rect_transformation(body, w, h, scale = 1.25):
    """
    w, h : image input shape
    """
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt
    # # Expands pose rect with marging used during training.
    # node {
    #   calculator: "RectTransformationCalculator"
    #   input_stream: "NORM_RECT:raw_roi"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "roi"
    #   options: {
    #     [mediapipe.RectTransformationCalculatorOptions.ext] {
    # Version 0831:
    #       scale_x: 1.5
    #       scale_y: 1.5
    # Version 084:
    #       scale_x: 1.25
    #       scale_y: 1.25
    #       square_long: true
    #     }
    #   }
    # }
    scale_x = scale
    scale_y = scale
    shift_x = 0
    shift_y = 0

    width = body.rect_w
    height = body.rect_h
    rotation = body.rotation
    if rotation == 0:
        body.rect_x_center_a = (body.rect_x_center + width * shift_x) * w
        body.rect_y_center_a = (body.rect_y_center + height * shift_y) * h
    else:
        x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation))
        y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation))
        body.rect_x_center_a = body.rect_x_center*w + x_shift
        body.rect_y_center_a = body.rect_y_center*h + y_shift

    # square_long: true
    long_side = max(width * w, height * h)
    body.rect_w_a = long_side * scale_x
    body.rect_h_a = long_side * scale_y
    body.rect_points = rotated_rect_to_points(body.rect_x_center_a, body.rect_y_center_a, body.rect_w_a, body.rect_h_a, body.rotation)

def warp_rect_img(rect_points, img, w, h):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (w, 0), (w, h)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (w, h))

def distance(a, b):
    """
    a, b: 2 points in 3D (x,y,z)
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

#
def find_isp_scale_params(size, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    is_height : boolean that indicates if the value is the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288
    if size < 288:
        size = 288

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = 1080
        other = 1920
    else:
        reference = 1920
        other = 1080
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

#
# Filtering
#

class LandmarksSmoothingFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/calculators/util/landmarks_smoothing_calculator.cc

    frequency, min_cutoff, beta, derivate_cutoff:
                See class OneEuroFilter description.
    min_allowed_object_scale:
                If calculated object scale is less than given value smoothing will be
                disabled and landmarks will be returned as is. Default=1e-6
    disable_value_scaling:
                Disable value scaling based on object size and use `1.0` instead.
                If not disabled, value scale is calculated as inverse value of object
                size. Object size is calculated as maximum side of rectangular bounding
                box of the object in XY plane. Default=False
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                min_allowed_object_scale=1e-6,
                disable_value_scaling=False
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.min_allowed_object_scale = min_allowed_object_scale
        self.disable_value_scaling = disable_value_scaling
        self.init = True

    @staticmethod
    def get_object_scale(landmarks):
        # Estimate object scale to use its inverse value as velocity scale for
        # RelativeVelocityFilter. If value will be too small (less than
        # `options_.min_allowed_object_scale`) smoothing will be disabled and
        # landmarks will be returned as is.
        # Object scale is calculated as average between bounding box width and height
        # with sides parallel to axis.
        min_xy = np.min(landmarks[:,:2], axis=0)
        max_xy = np.max(landmarks[:,:2], axis=0)
        return np.mean(max_xy - min_xy)

    def apply(self, landmarks, timestamp, object_scale=0):
        # object_scale: in practice, we use the size of the rotated rectangle region.rect_w_a=region.rect_h_a

        # Initialize filters
        if self.init:
            self.filters = OneEuroFilter(self.frequency, self.min_cutoff, self.beta, self.derivate_cutoff)
            self.init = False

        # Get value scale as inverse value of the object scale.
        # If value is too small smoothing will be disabled and landmarks will be
        # returned as is.
        if self.disable_value_scaling:
            value_scale = 1
        else:
            object_scale = object_scale if object_scale else self.get_object_scale(landmarks)
            if object_scale < self.min_allowed_object_scale:
                return landmarks
            value_scale = 1 / object_scale

        return self.filters.apply(landmarks, value_scale, timestamp)

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def reset(self):
        self.init = True

class OneEuroFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/one_euro_filter.cc
    Paper: https://cristal.univ-lille.fr/~casiez/1euro/

    frequency:
                Frequency of incoming frames defined in seconds. Used
                only if can't be calculated from provided events (e.g.
                on the very first frame). Default=30
    min_cutoff:
                Minimum cutoff frequency. Start by tuning this parameter while
                keeping `beta=0` to reduce jittering to the desired level. 1Hz
                (the default value) is a a good starting point.
    beta:
                Cutoff slope. After `min_cutoff` is configured, start
                increasing `beta` value to reduce the lag introduced by the
                `min_cutoff`. Find the desired balance between jittering and lag. Default=0
    derivate_cutoff:
                Cutoff frequency for derivate. It is set to 1Hz in the
                original algorithm, but can be turned to further smooth the
                speed (i.e. derivate) on the object. Default=1
    '''
    def __init__(self,
                frequency=30,
                min_cutoff=1,
                beta=0,
                derivate_cutoff=1,
                ):
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self.x = LowPassFilter(self.get_alpha(min_cutoff))
        self.dx = LowPassFilter(self.get_alpha(derivate_cutoff))
        self.last_timestamp = 0

    def get_alpha(self, cutoff):
        '''
        te = 1.0 / self.frequency
        tau = 1.0 / (2 * Math.PI * cutoff)
        result = 1 / (1.0 + (tau / te))
        '''
        return 1.0 / (1.0 + (self.frequency / (2 * pi * cutoff)))

    def apply(self, value, value_scale, timestamp):
        '''
        Applies filter to the value.
        timestamp in s associated with the value (for instance,
        timestamp of the frame where you got value from).
        '''
        if self.last_timestamp >= timestamp:
            # Results are unpreditable in this case, so nothing to do but return same value.
            return value

        # Update the sampling frequency based on timestamps.
        if self.last_timestamp != 0 and timestamp != 0:
            self.frequency = 1 / (timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Estimate the current variation per second.
        if self.x.has_last_raw_value():
            dvalue = (value - self.x.last_raw_value()) * value_scale * self.frequency
        else:
            dvalue = 0
        edvalue = self.dx.apply_with_alpha(dvalue, self.get_alpha(self.derivate_cutoff))

        # Use it to update the cutoff frequency
        cutoff = self.min_cutoff + self.beta * np.abs(edvalue)

        # filter the given value.
        return self.x.apply_with_alpha(value, self.get_alpha(cutoff))

class LowPassFilter:
    '''
    Adapted from: https://github.com/google/mediapipe/blob/master/mediapipe/util/filtering/low_pass_filter.cc
    Note that 'value' can be a numpy array
    '''
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.initialized = False

    def apply(self, value):
        if self.initialized:
            # Regular lowpass filter.
            # result = alpha * value + (1 - alpha) * stored_value;
            result = self.alpha * value + (1 - self.alpha) * self.stored_value
        else:
            result = value
            self.initialized = True
        self.raw_value = value
        self.stored_value = result
        return result

    def apply_with_alpha(self, value, alpha):
        self.alpha = alpha
        return self.apply(value)

    def has_last_raw_value(self):
        return self.initialized

    def last_raw_value(self):
        return self.raw_value

    def last_value(self):
        return self.stored_value

    def reset(self):
        self.initialized = False


