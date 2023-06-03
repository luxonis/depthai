from typing import List, Union, Dict

import depthai as dai
import numpy as np

from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.classes.packets import HandTrackerPacket, RotatedDetectionPacket
from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_seq_sync import XoutSeqSync
from depthai_sdk.visualize.visualizer import Visualizer
import depthai_sdk.components.hand_tracker.mediapipe_utils as mpu
import marshal
from depthai_sdk.visualize.bbox import BoundingBox
from math import sin, cos, atan2, pi, degrees, floor, dist
try:
    import cv2
except ImportError:
    cv2 = None

class XoutHandTracker(XoutSeqSync, XoutFrames):
    def __init__(self,
                 script_stream: StreamXout,
                 frames: StreamXout,
                 component: 'HandTrackerComponent'
                 ):
        self.script_stream = script_stream

        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, script_stream])
        self.pad_h = component.pad_h
        self.pad_w = component.pad_w
        self.name = 'HandTracker'

    def visualize(self, packet: HandTrackerPacket):
        print('Visualization of HandTracker is not supported')

    def xstreams(self) -> List[StreamXout]:
        return [self.script_stream, self.frames]

    def package(self, msgs: Dict):
        frame = msgs['0_preview'].getCvFrame()

        res = marshal.loads(msgs['host_host'].getData())
        hands = []
        pre_det_bb = BoundingBox().resize_to_aspect_ratio(frame.shape, (1, 1), resize_mode='letterbox')
        for i in range(len(res.get("lm_score",[]))):
            hand = self.extract_hand_data(res, i, frame, pre_det_bb)
            hands.append(hand)

        packet = HandTrackerPacket(name=self.name, hands=hands, color_frame=frame)
        self.queue.put(packet, block=False)

    def extract_hand_data(self, res, hand_idx, frame: np.ndarray, bb: BoundingBox):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        hand = mpu.HandRegion()

        # center = bb.map_point(res["rect_center_x"][hand_idx], res["rect_center_y"][hand_idx]).denormalize(frame.shape)
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * frame_width # center[0]
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * frame_height # center[1]
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * frame_width
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation)
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1,3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * frame_width).reshape(-1,2).astype(np.int32)

        if res.get("xyz", None) is not None:
            hand.xyz = np.array(res["xyz"][hand_idx])
            hand.xyz_zone = res["xyz_zone"][hand_idx]

        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            hand.landmarks[:,1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:,0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # World landmarks
        if res.get("world_lms", None) is not None:
            try:
                hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)
            except:
                pass

        # if self.use_gesture: mpu.recognize_gesture(hand)

        return hand

THRESHOLD = 0.5
class XoutPalmDetection(XoutSeqSync, XoutFrames):
    def __init__(self,
                 frames: StreamXout,
                 nn_results: StreamXout,
                 hand_tracker: 'HandTrackerComponent'
                 ):
        self.frames = frames
        self.nn_results = nn_results

        # self._decode_fn = None
        # self._labels = False
        self.size = hand_tracker.pd_size
        self.resize_mode = hand_tracker.pd_resize_mode

        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, nn_results])

    def xstreams(self) -> List[StreamXout]:
        return [self.nn_results, self.frames]

    def normalize_radians(self, angle):
        return angle - 2 * pi * floor((angle + pi) / (2 * pi))

    def visualize(self, packet: RotatedDetectionPacket):
        """
        Palm detection visualization callback
        """
        frame = packet.frame
        pre_det_bb = BoundingBox().resize_to_aspect_ratio(frame.shape, self.size, resize_mode=self.resize_mode)
        for rr, corners in zip(packet.rotated_rects, packet.bb_corners):
            # print(rr.center.x, rr.center.y, rr.size.width, rr.size.height, rr.angle)
            corner_points = [pre_det_bb.map_point(*p).denormalize(frame.shape) for p in corners]
            cv2.polylines(frame, [np.array(corner_points)], True, (127,255,0), 2, cv2.LINE_AA)
            for corner in corner_points:
                cv2.circle(frame, center=corner, radius=2, color=(0,127,255), thickness=-1)

            center = pre_det_bb.map_point(rr.center.x, rr.center.y).denormalize(frame.shape)
            cv2.circle(frame, center=center, radius=3, color=(255,127,0), thickness=-1)
        cv2.imshow('PD', frame)

    def package(self, msgs: Dict):
        pd_nndata: dai.NNData = msgs[self.nn_results.name]
        imgFrame: dai.ImgFrame = msgs[self.frames.name]
        vals = pd_nndata.getLayerFp16("result")

        # detection is list of Nx8 float
        # Looping the detection multiple times to obtain data for all hands
        rrs = []
        for i in range(0, len(vals), 8):
            pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = vals[i:i+8]
            if pd_score >= THRESHOLD and box_size > 0:
                # scale_center_x = sqn_scale_x - sqn_rr_center_x
                # scale_center_y = sqn_scale_y - sqn_rr_center_y
                kp02_x = kp2_x - kp0_x
                kp02_y = kp2_y - kp0_y
                sqn_rr_size = 2.9 * box_size
                rotation = 0.5 * pi - atan2(-kp02_y, kp02_x)
                rotation = self.normalize_radians(rotation)
                sqn_rr_center_x = box_x + 0.5*box_size*sin(rotation)
                sqn_rr_center_y = box_y - 0.5*box_size*cos(rotation)

                rr = dai.RotatedRect()
                rr.center.x = sqn_rr_center_x
                rr.center.y = sqn_rr_center_y
                rr.size.width = sqn_rr_size
                rr.size.height = sqn_rr_size
                rr.angle = degrees(rotation)
                rrs.append(rr)

        packet = RotatedDetectionPacket(
            name = self.get_packet_name(),
            msg = imgFrame,
            rotated_rects=rrs,
        )

        self.queue.put(packet, block=False)
