import cv2
import numpy as np

def get_tensor_output(nnet_packet, idx):
    np_array = nnet_packet.get_tensor(idx)
    np_array.dtype = np.float16
    return np_array

def get_tensor_outputs_list(nnet_packet):
    array = []
    size = nnet_packet.getTensorsSize()
    for i in range(size):
        tensor = get_tensor_output(nnet_packet, i)
        array.append(tensor)
    return array

def get_tensor_outputs_dict(nnet_packet):

    outputs_dict = nnet_packet.getOutputs()
    for array in outputs_dict:
        np_array = outputs_dict[array]
        #TODO find a way to return array as fp16
        np_array.dtype = np.float16
    return outputs_dict