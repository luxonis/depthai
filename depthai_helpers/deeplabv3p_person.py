import cv2
import numpy as np

class_colors = [[0,0,0],  [0,255,0]]
class_colors = np.asarray(class_colors, dtype=np.uint8)
output_colors = None

def decode_deeplabv3p(nnet_packet, **kwargs):
    output_tensor = nnet_packet.get_tensor(0)
    output = output_tensor[0][0]
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame, **kwargs):
    if type(output_colors) is not list:
        frame = cv2.addWeighted(frame,1, output_colors,0.2,0)
    return frame

