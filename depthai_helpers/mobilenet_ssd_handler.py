import cv2
import numpy as np
from datetime import datetime

def decode_mobilenet_ssd(nnet_packet, **kwargs):
    config = kwargs['config']
    detections = []
    # the result of the MobileSSD has detection rectangles (here: entries), and we can iterate through them
    for _, e in enumerate(nnet_packet.entries()):
        # for MobileSSD entries are sorted by confidence
        # {id == -1} or {confidence == 0} is the stopper (special for OpenVINO models and MobileSSD architecture)
        if e[0]['id'] == -1.0 or e[0]['confidence'] == 0.0:
            break
        # save entry for further usage (as image package may arrive not the same time as nnet package)
        # the lower confidence threshold - the more we get false positives
        if e[0]['confidence'] > config['depth']['confidence_threshold']:
            detections.append(e)
    return detections


def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

def average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = int((pt2[0] - pt1[0]) * factor / 2)
    y_shift = int((pt2[1] - pt1[1]) * factor / 2)
    avg_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    avg_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return avg_pt1, avg_pt2

def show_mobilenet_ssd(entries_prev, frame, **kwargs):
    is_depth = 'nn2depth' in kwargs
    if is_depth:
        nn2depth = kwargs['nn2depth']
    config = kwargs['config']
    labels = kwargs['labels']
    img_h = frame.shape[0]
    img_w = frame.shape[1]

    last_detected = datetime.now()
    # iterate through pre-saved entries & draw rectangle & text on image:
    iteration = 0
    for e in entries_prev:
        if is_depth:
            pt1 = nn_to_depth_coord(e[0]['left'],  e[0]['top'], nn2depth)
            pt2 = nn_to_depth_coord(e[0]['right'], e[0]['bottom'], nn2depth)
            color = (255, 0, 0) # bgr
            avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2, config['depth']['padding_factor'])
            cv2.rectangle(frame, avg_pt1, avg_pt2, color)
            color = (255, 255, 255) # bgr
        else:
            pt1 = int(e[0]['left']  * img_w), int(e[0]['top']    * img_h)
            pt2 = int(e[0]['right'] * img_w), int(e[0]['bottom'] * img_h)
            color = (0, 0, 255) # bgr

        x1, y1 = pt1
        x2, y2 = pt2

        cv2.rectangle(frame, pt1, pt2, color)
        # Handles case where TensorEntry object label is out if range
        if e[0]['label'] > len(labels):
            print("Label index=",e[0]['label'], "is out of range. Not applying text to rectangle.")
        else:
            pt_t1 = x1, y1 + 20
            cv2.putText(frame, labels[int(e[0]['label'])], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            pt_t2 = x1, y1 + 40
            cv2.putText(frame, '{:.2f}'.format(100*e[0]['confidence']) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            if config['ai']['calc_dist_to_bb']:
                pt_t3 = x1, y1 + 60
                cv2.putText(frame, 'x:' '{:7.3f}'.format(e[0]['distance_x']) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                pt_t4 = x1, y1 + 80
                cv2.putText(frame, 'y:' '{:7.3f}'.format(e[0]['distance_y']) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                pt_t5 = x1, y1 + 100
                cv2.putText(frame, 'z:' '{:7.3f}'.format(e[0]['distance_z']) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
            # Second-stage NN
            if iteration == 0: # For now we run second-stage only on first detection
                if 'landmarks-regression-retail-0009' in config['ai']['blob_file2']:
                    # Decode
                    landmarks = []
                    for i in range(len(e[1])):
                        landmarks.append(e[1][i])
                    landmarks = list(zip(*[iter(landmarks)]*2))
                    # Show
                    bb_w = x2 - x1
                    bb_h = y2 - y1
                    for i in landmarks:
                        try:
                            x = x1 + int(i[0]*bb_w)
                            y = y1 + int(i[1]*bb_h)
                        except:
                            continue
                        cv2.circle(frame, (x,y), 4, (255, 0, 0))
                if 'emotions-recognition-retail-0003' in config['ai']['blob_file2']:
                    # Decode
                    emotion_data = []
                    for i in range(len(e[1])):
                        emotion_data.append(e[1][i])
                    # Show
                        e_states = {
                            0 : "neutral",
                            1 : "happy",
                            2 : "sad",
                            3 : "surprise",
                            4 : "anger"
                        }
                    pt_t3 = x2-50, y2-10
                    max_confidence = max(emotion_data)
                    if(max_confidence > 0.7):
                        emotion = e_states[np.argmax(emotion_data)]
                        if (datetime.now() - last_detected).total_seconds() < 100:
                            cv2.putText(frame, emotion, pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 2)
        iteration += 1
    return frame

