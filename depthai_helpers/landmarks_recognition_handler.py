import cv2
import numpy as np
import json

def decode_landmarks_recognition(nnet_packet, **kwargs):
    landmarks_tensor = nnet_packet.get_tensor(0)
    landmarks = []

    landmarks_tensor = landmarks_tensor.reshape(landmarks_tensor.shape[1])    
    landmarks = list(zip(*[iter(landmarks_tensor)]*2))
    return landmarks

def decode_landmarks_recognition_json(nnet_packet, **kwargs):
    landmarks = decode_landmarks_recognition(nnet_packet, **kwargs)
    return json.dumps(landmarks)

def show_landmarks_recognition(landmarks, frame, **kwargs):
    img_h = frame.shape[0]
    img_w = frame.shape[1]

    if len(landmarks) != 0:
        for i in landmarks:
            x = int(i[0]*img_h)
            y = int(i[1]*img_w)
            # # print(x,y)
            cv2.circle(frame, (x,y), 3, (0, 0, 255))

    frame = cv2.resize(frame, (300, 300))

    return frame
