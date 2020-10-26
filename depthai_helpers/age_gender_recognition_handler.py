import cv2
import numpy as np
import json

def decode_age_gender_recognition(nnet_packet, **kwargs):
    config = kwargs['config']

    output_list = nnet_packet.getOutputsList()

    age_out = output_list[0]
    age = age_out[0,0,0,0]
    gender_out = output_list[1]
    female_conf = gender_out[0,0,0,0]
    male_conf = gender_out[0,1,0,0]

    detection = None

    conf_thr = 0.5
    if female_conf > conf_thr or male_conf > conf_thr:
        gender = "male"
        if female_conf > male_conf:
            gender = "female"
        age = (int)(age*100)
        detection = dict(gender=gender, age=age)
    
    return detection

def decode_age_gender_recognition_json(nnet_packet, **kwargs):
    detections = decode_age_gender_recognition(nnet_packet, **kwargs)
    return json.dumps(detections)

def show_age_gender_recognition(decoded_nn, frame, **kwargs):

    if decoded_nn is not None:
        gender = decoded_nn["gender"]
        age = decoded_nn["age"]
        cv2.putText(frame, "Age: " + str(age), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "G: " + str(gender), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    frame = cv2.resize(frame, (300, 300))
    return frame
