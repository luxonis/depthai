import torch
import torch.nn as nn
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))
import numpy as np
from math import pi



detection_input_length = 224

class DivideBy255(nn.Module):
    def __init__(self):
        super(DivideBy255, self).__init__()

    def forward(self, x):
        return x / 255.0

def test():

    model = DivideBy255()
    
    X = torch.ones(1, 3, 256, 256, dtype=torch.float)
    result = model(X)
    print(result)

def export_onnx():
    """
    Exports the model to an ONNX file.
    """
    model = DivideBy255()
    X = torch.randn(1, 3, 256, 256,  dtype=torch.float)
    onnx_name = "DivideBy255.onnx"

    print(f"Generating {onnx_name}")
    torch.onnx.export(
        model,
        (X),
        onnx_name,
        opset_version=10,
        do_constant_folding=True,
        # verbose=True,
        # input_names=['Identity_1', 'Identity'],
        output_names=['input_1']
    )

if __name__ == "__main__":

    test()
    export_onnx()