import cv2
import numpy as np

def show_tracklets(tracklets, frame, labels):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]

    # iterate through pre-saved entries & draw rectangle & text on image:
    tracklet_nr = tracklets.getNrTracklets()

    for i in range(tracklet_nr):
        tracklet        = tracklets.getTracklet(i)
        left_coord      = tracklet.getLeftCoord()
        top_coord       = tracklet.getTopCoord()
        right_coord     = tracklet.getRightCoord()
        bottom_coord    = tracklet.getBottomCoord()
        tracklet_id     = tracklet.getId()
        tracklet_label  = labels[tracklet.getLabel()]
        tracklet_status = tracklet.getStatus()

        # print("left: {0} top: {1} right: {2}, bottom: {3}, id: {4}, label: {5}, status: {6} "\
        #     .format(left_coord, top_coord, right_coord, bottom_coord, tracklet_id, tracklet_label, tracklet_status))
        
        pt1 = left_coord,  top_coord
        pt2 = right_coord,  bottom_coord
        color = (255, 0, 0) # bgr
        cv2.rectangle(frame, pt1, pt2, color)

        middle_pt = (int)(left_coord + (right_coord - left_coord)/2), (int)(top_coord + (bottom_coord - top_coord)/2)
        cv2.circle(frame, middle_pt, 0, color, -1)
        cv2.putText(frame, "ID {0}".format(tracklet_id), middle_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        x1, y1 = left_coord,  bottom_coord


        pt_t1 = x1, y1 - 40
        cv2.putText(frame, tracklet_label, pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pt_t2 = x1, y1 - 20
        cv2.putText(frame, tracklet_status, pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        
    return frame
