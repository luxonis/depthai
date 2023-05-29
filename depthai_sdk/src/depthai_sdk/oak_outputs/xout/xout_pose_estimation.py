from typing import List, Union, Dict
import time
import depthai as dai
import numpy as np

from depthai_sdk.classes.enum import ResizeMode
from depthai_sdk.classes.packets import BodyPosePacket, RotatedDetectionPacket
from depthai_sdk.oak_outputs.xout.xout_base import StreamXout
from depthai_sdk.oak_outputs.xout.xout_frames import XoutFrames
from depthai_sdk.oak_outputs.xout.xout_seq_sync import XoutSeqSync
from depthai_sdk.visualize.visualizer import Visualizer
import depthai_sdk.components.pose_estimation.mediapipe_utils as mpu
import marshal
from depthai_sdk.visualize.bbox import BoundingBox
from math import sin, cos, atan2, pi, hypot, degrees, floor
try:
    import cv2
except ImportError:
    cv2 = None

class XoutBlazepose(XoutSeqSync, XoutFrames):
    def __init__(self,
                 script_stream: StreamXout,
                 frames: StreamXout,
                 component: 'BlazeposeComponent'
                 ):
        self.script_stream = script_stream
        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, script_stream])
        self.pad_h = component.pad_h
        self.pad_w = component.pad_w
        self.name = 'Blazepose'

        self.smoothing = component.smoothing
        self.nb_kps = component.nb_kps
        self.filter_landmarks = component.filter_landmarks
        self.filter_landmarks_aux = component.filter_landmarks_aux
        self.filter_landmarks_world = component.filter_landmarks_world
        self.lm_input_length = component.lm_input_length
        self.presence_threshold = component.presence_threshold
        self.lm_score_thresh = component.lm_score_thresh
        self.xyz = component.xyz
        self.filter_xyz = component.filter_xyz

    def visualize(self, packet: BodyPosePacket):
        print('Visualization of Blazepose is not supported')

    def xstreams(self) -> List[StreamXout]:
        return [self.script_stream, self.frames]

    def package(self, msgs: Dict):
        frame = msgs[self.frames.name].getCvFrame()
        width = frame.shape[1]
        height = frame.shape[0]

        res = marshal.loads(msgs[self.script_stream.name].getData())
        pre_det_bb = BoundingBox().resize_to_aspect_ratio(frame.shape, (1, 1), resize_mode='letterbox')
        body = None

        if res["type"] != 0 and res["lm_score"] > self.lm_score_thresh:
            body = mpu.Body()
            body.rect_x_center_a = res["rect_center_x"] * width
            body.rect_y_center_a = res["rect_center_y"] * width
            body.rect_w_a = body.rect_h_a = res["rect_size"] * width
            body.rotation = res["rotation"]
            body.rect_points = mpu.rotated_rect_to_points(
                body.rect_x_center_a, body.rect_y_center_a, body.rect_w_a, body.rect_h_a, body.rotation)
            body.lm_score = res["lm_score"]
            self.lm_postprocess(body, res['lms'], res['lms_world'])
            if self.xyz:
                if res['xyz_ref'] == 0:
                    body.xyz_ref = None
                else:
                    if res['xyz_ref'] == 1:
                        body.xyz_ref = "mid_hips"
                    else:  # res['xyz_ref'] == 2:
                        body.xyz_ref = "mid_shoulders"
                    body.xyz = np.array(res["xyz"])
                    if self.smoothing:
                        body.xyz = self.filter_xyz.apply(body.xyz)
                    body.xyz_zone = np.array(res["xyz_zone"])
                    body.xyz_ref_coords_pixel = np.mean(
                        body.xyz_zone.reshape((2, 2)), axis=0)

        else:
            body = None
            if self.smoothing:
                self.filter_landmarks.reset()
                self.filter_landmarks_aux.reset()
                self.filter_landmarks_world.reset()
                # if self.xyz:
                #     self.filter_xyz.reset()


        packet = BodyPosePacket(name=self.name, body=body, color_frame=frame)
        self.queue.put(packet, block=False)

    def lm_postprocess(self, body, lms, lms_world):
        # lms : landmarks sent by Manager script node to host (list of 39*5 elements for full body or 31*5 for upper body)
        lm_raw = np.array(lms).reshape(-1, 5)
        # Each keypoint have 5 information:
        # - X,Y coordinates are local to the body of
        # interest and range from [0.0, 255.0].
        # - Z coordinate is measured in "image pixels" like
        # the X and Y coordinates and represents the
        # distance relative to the plane of the subject's
        # hips, which is the origin of the Z axis. Negative
        # values are between the hips and the camera;
        # positive values are behind the hips. Z coordinate
        # scale is similar with X, Y scales but has different
        # nature as obtained not via human annotation, by
        # fitting synthetic data (GHUM model) to the 2D
        # annotation.
        # - Visibility, after user-applied sigmoid denotes the
        # probability that a keypoint is located within the
        # frame and not occluded by another bigger body
        # part or another object.
        # - Presence, after user-applied sigmoid denotes the
        # probability that a keypoint is located within the
        # frame.

        # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
        lm_raw[:, :3] /= self.lm_input_length
        # Apply sigmoid on visibility and presence (if used later)
        body.visibility = 1 / (1 + np.exp(-lm_raw[:, 3]))
        body.presence = 1 / (1 + np.exp(-lm_raw[:, 4]))

        # body.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
        body.norm_landmarks = lm_raw[:, :3]
        # Now calculate body.landmarks = the landmarks in the image coordinate system (in pixel) (body.landmarks)
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        # body.rect_points[0] is left bottom point and points going clockwise!
        dst = np.array([(x, y)
                       for x, y in body.rect_points[1:]], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps, :2], axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat))

        # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
        # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
        lm_z = body.norm_landmarks[:self.nb_kps, 2:3] * body.rect_w_a / 4
        lm_xyz = np.hstack((lm_xy, lm_z))

        # World landmarks are predicted in meters rather than in pixels of the image
        # and have origin in the middle of the hips rather than in the corner of the
        # pose image (cropped with given rectangle). Thus only rotation (but not scale
        # and translation) is applied to the landmarks to transform them back to
        # original  coordinates.
        body.landmarks_world = np.array(lms_world).reshape(-1, 3)
        sin_rot = sin(body.rotation)
        cos_rot = cos(body.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        body.landmarks_world[:, :2] = np.dot(
            body.landmarks_world[:, :2], rot_m)

        if self.smoothing:
            timestamp = time.perf_counter()
            object_scale = body.rect_w_a
            lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(
                lm_xyz[:self.nb_kps], timestamp, object_scale)
            lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(
                lm_xyz[self.nb_kps:], timestamp, object_scale)
            body.landmarks_world = self.filter_landmarks_world.apply(
                body.landmarks_world, timestamp)

        body.landmarks = lm_xyz.astype(np.int32)
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            body.landmarks[:, 1] -= self.pad_h
            for i in range(len(body.rect_points)):
                body.rect_points[i][1] -= self.pad_h
        # if self.pad_w > 0:
        #     body.landmarks[:,0] -= self.pad_w
        #     for i in range(len(body.rect_points)):
        #         body.rect_points[i][0] -= self.pad_w




POSE_DETECTION_THRESHOLD = 0.5
class XoutPoseDetections(XoutSeqSync, XoutFrames):
    def __init__(self,
                 frames: StreamXout,
                 nn_results: StreamXout,
                 component: 'BlazeposeComponent'
                 ):
        self.frames = frames
        self.nn_results = nn_results

        self.size = (component.pd_input_length, component.pd_input_length)
        self.resize_mode = component.resize_mode
        self.rect_transf_scale = component.rect_transf_scale
        self.pad_h = component.pad_h

        # self._decode_fn = None
        # self._labels = False

        XoutFrames.__init__(self, frames)
        XoutSeqSync.__init__(self, [frames, nn_results])

    def xstreams(self) -> List[StreamXout]:
        return [self.nn_results, self.frames]

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
        frame = imgFrame.getCvFrame()
        width = frame.shape[1]
        height = frame.shape[0]

        detection = pd_nndata.getLayerFp16("result")
        print("Manager received pd result: "+str(detection))

        # TODO: debug

        # pd_score, sqn_rr_center_x, sqn_rr_center_y, sqn_scale_x, sqn_scale_y = detection
        # if pd_score < POSE_DETECTION_THRESHOLD:
        #     print("Pose detection score is too low: "+str(pd_score))
        #     return
        # scale_center_x = sqn_scale_x - sqn_rr_center_x
        # scale_center_y = sqn_scale_y - sqn_rr_center_y
        # sqn_rr_size = 2 * self.rect_transf_scale * hypot(scale_center_x, scale_center_y)
        # rotation = 0.5 * pi - atan2(-scale_center_y, scale_center_x)
        # rotation = rotation - 2 * pi *floor((rotation + pi) / (2 * pi))

        # # Tell pre_lm_manip how to crop body region
        # rr = dai.RotatedRect()
        # rr.center.x    = sqn_rr_center_x
        # rr.center.y    = (sqn_rr_center_y * height - self.pad_h) / height
        # rr.size.width  = sqn_rr_size
        # rr.size.height = sqn_rr_size
        # rr.angle       = degrees(rotation)

        # packet = RotatedDetectionPacket(
        #     name = self.get_packet_name(),
        #     msg = imgFrame,
        #     rotated_rects=[rr],
        # )
        # self.queue.put(packet, block=False)


class XoutBlazeposePassthrough(XoutBlazepose):
    def lm_postprocess(self, body, lms, lms_world):

        # lms : landmarks sent by Manager script node to host (list of 39*5 elements for full body or 31*5 for upper body)
        lm_raw = np.array(lms).reshape(-1, 5)
        body.lm_raw = lm_raw.copy()
        # Each keypoint have 5 information:
        # - X,Y coordinates are local to the body of
        # interest and range from [0.0, 255.0].
        # - Z coordinate is measured in "image pixels" like
        # the X and Y coordinates and represents the
        # distance relative to the plane of the subject's
        # hips, which is the origin of the Z axis. Negative
        # values are between the hips and the camera;
        # positive values are behind the hips. Z coordinate
        # scale is similar with X, Y scales but has different
        # nature as obtained not via human annotation, by
        # fitting synthetic data (GHUM model) to the 2D
        # annotation.
        # - Visibility, after user-applied sigmoid denotes the
        # probability that a keypoint is located within the
        # frame and not occluded by another bigger body
        # part or another object.
        # - Presence, after user-applied sigmoid denotes the
        # probability that a keypoint is located within the
        # frame.

        # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
        lm_raw[:, :3] /= self.lm_input_length
        # Apply sigmoid on visibility and presence (if used later)
        body.visibility = 1 / (1 + np.exp(-lm_raw[:, 3]))
        body.presence = 1 / (1 + np.exp(-lm_raw[:, 4]))

        # body.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
        body.norm_landmarks = lm_raw[:, :3]
        # Now calculate body.landmarks = the landmarks in the image coordinate system (in pixel) (body.landmarks)
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        # body.rect_points[0] is left bottom point and points going clockwise!
        dst = np.array([(x, y)
                       for x, y in body.rect_points[1:]], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps, :2], axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat))

        # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
        # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
        lm_z = body.norm_landmarks[:self.nb_kps, 2:3] * body.rect_w_a / 4
        lm_xyz = np.hstack((lm_xy, lm_z))

        # World landmarks are predicted in meters rather than in pixels of the image
        # and have origin in the middle of the hips rather than in the corner of the
        # pose image (cropped with given rectangle). Thus only rotation (but not scale
        # and translation) is applied to the landmarks to transform them back to
        # original  coordinates.
        body.landmarks_world = np.array(lms_world).reshape(-1, 3)
        sin_rot = sin(body.rotation)
        cos_rot = cos(body.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        body.landmarks_world[:, :2] = np.dot(
            body.landmarks_world[:, :2], rot_m)

        if self.smoothing:
            timestamp = time.perf_counter()
            object_scale = body.rect_w_a
            lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(
                lm_xyz[:self.nb_kps], timestamp, object_scale)
            lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(
                lm_xyz[self.nb_kps:], timestamp, object_scale)
            body.landmarks_world = self.filter_landmarks_world.apply(
                body.landmarks_world, timestamp)

        body.landmarks = lm_xyz.astype(np.int32)
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
