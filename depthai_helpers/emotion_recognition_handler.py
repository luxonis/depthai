import cv2
import numpy as np
import json

def decode_emotion_recognition(nnet_packet, **kwargs):
    em_tensor = nnet_packet.get_tensor(0)
    detections = []
    for i in em_tensor[0]:
        detections.append(i[0][0])
    return detections

def decode_emotion_recognition_json(nnet_packet, **kwargs):
    detections = decode_emotion_recognition(nnet_packet, **kwargs)
    return json.dumps(detections)

def show_emotion_recognition(entries_prev, frame, **kwargs):

    NN_metadata = kwargs['NN_json']
    labels = NN_metadata['mappings']['labels']

    if len(entries_prev) != 0:
        max_confidence = max(entries_prev)
        if(max_confidence > 0.7):
            emotion = labels[np.argmax(entries_prev)]
            cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.resize(frame, (300, 300))

    return frame
