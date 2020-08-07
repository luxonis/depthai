import cv2
import numpy as np
from datetime import datetime
from depthai_helpers.tensor_utils import get_tensor_output, get_tensor_outputs_list, get_tensor_outputs_dict

def decode_mobilenet_ssd(nnet_packet, **kwargs):
    return nnet_packet


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

def show_mobilenet_ssd(nnet_packet, frame, **kwargs):
    res = get_tensor_output(nnet_packet, 0)

    is_depth = 'nn2depth' in kwargs
    if is_depth:
        nn2depth = kwargs['nn2depth']
    config = kwargs['config']
    labels = kwargs['labels']
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    last_detected = datetime.now()
    # iterate through pre-saved entries & draw rectangle & text on image:
    iteration = 0
    for obj in res[0][0]:
        score = obj[2]
        # Draw only objects when probability more than specified threshold
        if score > config['depth']['confidence_threshold']:
            xmin = obj[3]
            ymin = obj[4]
            xmax = obj[5]
            ymax = obj[6]
            class_id = int(obj[1])
           

            # Draw box and label\class_id
            # color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            # det_label = labels[class_id] if labels else str(class_id)
            # cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
            #             cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if is_depth:
                pt1 = nn_to_depth_coord(xmin,  ymin, nn2depth)
                pt2 = nn_to_depth_coord(xmax, ymax, nn2depth)
                color = (255, 0, 0) # bgr
                avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2, config['depth']['padding_factor'])
                cv2.rectangle(frame, avg_pt1, avg_pt2, color)
                color = (255, 255, 255) # bgr
            else:
                pt1 = int(xmin  * frame_w), int(ymin    * frame_h)
                pt2 = int(xmax * frame_w), int(ymax * frame_h)
                color = (0, 0, 255) # bgr

            x1, y1 = pt1
            x2, y2 = pt2

            cv2.rectangle(frame, pt1, pt2, color)
            # Handles case where TensorEntry object label is out if range
            if class_id > len(labels):
                print("Label index=",class_id, "is out of range. Not applying text to rectangle.")
            else:
                pt_t1 = x1, y1 + 20
                cv2.putText(frame, labels[int(class_id)], pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                pt_t2 = x1, y1 + 40
                cv2.putText(frame, '{:.2f}'.format(100*score) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                if config['ai']['calc_dist_to_bb']:
                    distance_x = obj[7]
                    distance_y = obj[8]
                    distance_z = obj[9]
                    pt_t3 = x1, y1 + 60
                    cv2.putText(frame, 'x:' '{:7.3f}'.format(distance_x) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    pt_t4 = x1, y1 + 80
                    cv2.putText(frame, 'y:' '{:7.3f}'.format(distance_y) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

                    pt_t5 = x1, y1 + 100
                    cv2.putText(frame, 'z:' '{:7.3f}'.format(distance_z) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
                # Second-stage NN
                if iteration == 0: # For now we run second-stage only on first detection
                    if 'landmarks-regression-retail-0009' in config['ai']['blob_file2']:
                        landmark_tensor = get_tensor_output(nnet_packet, 1)
                        # Decode
                        landmarks = []
                        for i in landmark_tensor[0]:
                            landmarks.append(i[0][0])
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
                        em_tensor = get_tensor_output(nnet_packet, 1)
                        # Decode
                        emotion_data = []
                        for i in em_tensor[0]:
                            emotion_data.append(i[0][0])
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

