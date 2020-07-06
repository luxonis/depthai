import cv2
import numpy as np


def decode_emotion_recognition(nnet_packet, **kwargs):
    detections = []
    for i in range(len(nnet_packet.entries()[0][0])):
        detections.append(nnet_packet.entries()[0][0][i])
    return detections

def show_emotion_recognition(entries_prev, frame, **kwargs):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]
    e_states = kwargs['labels']

    if len(entries_prev) != 0:
        max_confidence = max(entries_prev)
        if(max_confidence > 0.7):
            emotion = e_states[np.argmax(entries_prev)]
            cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame = cv2.resize(frame, (300, 300))

    return frame