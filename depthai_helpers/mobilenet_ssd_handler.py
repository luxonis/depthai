import cv2
import json
import numpy as np
from datetime import datetime

def decode_mobilenet_ssd(nnet_packet, **kwargs):
    NN_metadata = kwargs['NN_json']
    output_format = NN_metadata['NN_config']['output_format']
    config = kwargs['config']

    detections_list = []
    if output_format == "detection":
        detections = nnet_packet.getDetectedObjects()
        for detection in detections:
            detection_dict = detection.get_dict()
            detections_list.append(detection_dict)

    else:
        confidence_threshold = NN_metadata['NN_config']['confidence_threshold']
        res = nnet_packet.get_tensor(0)
        decoding_lut = {"valid_detection" : 0,
            "label" : 1,
            "confidence" : 2,
            "x_min" : 3,
            "y_min" : 4,
            "x_max" : 5,
            "y_max" : 6 }
        # iterate through pre-saved entries & draw rectangle & text on image:
        for obj in res[0][0]:
            confidence = obj[decoding_lut["confidence"]]
            if obj[decoding_lut["valid_detection"]] == -1.0 or confidence == 0.0:
                break
            label = int(obj[decoding_lut["label"]])
            x_min = obj[decoding_lut["x_min"]]
            y_min = obj[decoding_lut["y_min"]]
            x_max = obj[decoding_lut["x_max"]]
            y_max = obj[decoding_lut["y_max"]]
            if confidence > confidence_threshold:
                det_dict = dict(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, label=label, confidence=confidence)
                detections_list.append(det_dict)


    stage2_detections=None
    # Second-stage NN
    if any(x in config['ai']['blob_file2'] for x in ['landmarks-regression-retail-0009', 'facial-landmarks-35-adas-0002'] ):
        landmarks_tensor = nnet_packet.get_tensor(1)
        landmarks = []

        landmarks_tensor = landmarks_tensor.reshape(landmarks_tensor.shape[1])    
        landmarks = list(zip(*[iter(landmarks_tensor)]*2))
        stage2_detections=landmarks
    if 'emotions-recognition-retail-0003' in config['ai']['blob_file2']:
        em_tensor = nnet_packet.get_tensor(1)
        # Decode
        emotion_data = []
        for i in em_tensor[0]:
            emotion_data.append(i[0][0])
        stage2_detections=emotion_data
    

    detections = dict(stage1=detections_list, stage2=stage2_detections)
    return detections

def decode_mobilenet_ssd_json(nnet_packet, **kwargs):
    convertList = []

    detections = decode_mobilenet_ssd(nnet_packet, **kwargs)

    for entry in detections:
        jsonConvertDict = {}
        jsonConvertDict["id"] = entry[0]["id"]
        jsonConvertDict["label"] = entry[0]["label"]
        jsonConvertDict["confidence"] = entry[0]["confidence"]
        jsonConvertDict["left"] = entry[0]["left"]
        jsonConvertDict["right"] = entry[0]["right"]
        jsonConvertDict["top"] = entry[0]["top"]
        jsonConvertDict["bottom"] = entry[0]["bottom"]
        jsonConvertDict["distance_x"] = entry[0]["distance_x"]
        jsonConvertDict["distance_y"] = entry[0]["distance_y"]
        jsonConvertDict["distance_z"] = entry[0]["distance_z"]
        convertList.append([jsonConvertDict])

    jsonStr = json.dumps(convertList)

    return jsonStr

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


def show_mobilenet_ssd(detections, frame, **kwargs):
    is_depth = 'nn2depth' in kwargs
    if is_depth:
        nn2depth = kwargs['nn2depth']
    config = kwargs['config']
    NN_metadata = kwargs['NN_json']
    try:
        labels = NN_metadata['mappings']['labels']
    except:
        labels = None
    
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    last_detected = datetime.now()
    # iterate through pre-saved entries & draw rectangle & text on image:
    for idx, detection in enumerate(detections["stage1"]):
        # print(detection)
        if is_depth:
            pt1 = nn_to_depth_coord(detection["x_min"], detection["y_min"], nn2depth)
            pt2 = nn_to_depth_coord(detection["x_max"], detection["y_max"], nn2depth)
            color = (255, 0, 0) # bgr
            avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2, config['depth']['padding_factor'])
            cv2.rectangle(frame, avg_pt1, avg_pt2, color)
            color = (255, 255, 255) # bgr
        else:
            pt1 = int(detection["x_min"]  * frame_w), int(detection["y_min"]    * frame_h)
            pt2 = int(detection["x_max"] * frame_w), int(detection["y_max"] * frame_h)
            color = (0, 0, 255) # bgr

        x1, y1 = pt1
        x2, y2 = pt2

        cv2.rectangle(frame, pt1, pt2, color)
        # Handles case where TensorEntry object label is out if range
        if labels is not None:
            label = labels[detection["label"]]
        else:
            label = str(detection["label"])

        pt_t1 = x1, y1 + 20
        cv2.putText(frame, label, pt_t1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        pt_t2 = x1, y1 + 40
        cv2.putText(frame, '{:.2f}'.format(100*detection["confidence"]) + ' %', pt_t2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        if config['ai']['calc_dist_to_bb']:
            pt_t3 = x1, y1 + 60
            cv2.putText(frame, 'x:' '{:7.3f}'.format(detection["depth_x"]) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            pt_t4 = x1, y1 + 80
            cv2.putText(frame, 'y:' '{:7.3f}'.format(detection["depth_y"]) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

            pt_t5 = x1, y1 + 100
            cv2.putText(frame, 'z:' '{:7.3f}'.format(detection["depth_z"]) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        
        # Second-stage NN
        if idx == 0: # For now we run second-stage only on first detection
            if any(x in config['ai']['blob_file2'] for x in ['landmarks-regression-retail-0009', 'facial-landmarks-35-adas-0002'] ):
                landmarks = detections["stage2"]
                # Show
                bb_w = x2 - x1
                bb_h = y2 - y1
                for i in landmarks:
                    x = x1 + int(i[0]*bb_w)
                    y = y1 + int(i[1]*bb_h)
                    cv2.circle(frame, (x,y), 4, (255, 0, 0))
            if 'emotions-recognition-retail-0003' in config['ai']['blob_file2']:
                # Decode
                emotion_data = detections["stage2"]
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
                if(max_confidence > 0.6):
                    emotion = e_states[np.argmax(emotion_data)]
                    if (datetime.now() - last_detected).total_seconds() < 100:
                        cv2.putText(frame, emotion, pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255, 0), 2)
    
    return frame
