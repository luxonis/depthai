import cv2
import numpy as np
from depthai_helpers.tensor_utils import *


def decode_age_gender_recognition(nnet_packet, **kwargs):
    return nnet_packet

def show_age_gender_recognition(entries_prev, frame, **kwargs):
    # img_h = frame.shape[0]
    # img_w = frame.shape[1]

    config = kwargs['config']

    output_list = get_tensor_outputs_list(entries_prev)

    age_out = output_list[0]
    age = age_out[0,0,0,0]
    gender_out = output_list[1]
    female_conf = gender_out[0,0,0,0]
    male_conf = gender_out[0,1,0,0]

    conf_thr = config['depth']['confidence_threshold']
    if female_conf > 0.8 or male_conf > 0.8:
        gender = "male"
        if female_conf > male_conf:
            gender = "female"
        age = (int)(age*100)
        cv2.putText(frame, "Age: " + str(age), (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "G: " + str(gender), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    frame = cv2.resize(frame, (300, 300))
    return frame
