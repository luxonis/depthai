import torch
import torch.nn as nn
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from mediapipe_utils import generate_blazepose_anchors
import numpy as np
from math import pi

anchors = generate_blazepose_anchors()
print(f"Nb anchors: {len(anchors)}")

detection_input_length = 224

class DetectionBestCandidate(nn.Module):
    def __init__(self):
        super(DetectionBestCandidate, self).__init__()
        # anchors shape is nb_anchorsx4 [x_center, y_center, width, height]
        # Here: width and height is always 1, so we keep just [x_center, y_center]
        self.anchors = torch.from_numpy(anchors[:,:2])
        self.plus_anchor_center = np.array([[1,0,0,0,1,0,1,0,1,0,1,0], [0,1,0,0,0,1,0,1,0,1,0,1]], dtype=np.float)
        self.plus_anchor_center = torch.from_numpy(self.plus_anchor_center)

    def forward(self, x, y):
        # x.shape: 1xnb_anchorsx1
        # y.shape: 1xnb_anchorsx12

        # decode_bboxes
        best_id = torch.argmax(x)
        score = x[0][best_id]
        score = torch.sigmoid(score) 
        bbox = y[0,best_id]
        bbox = bbox/detection_input_length + torch.mm(self.anchors[best_id].unsqueeze(0), self.plus_anchor_center)[0]

        # bbox : the 4 first elements describe the square bounding box around the head (not used for determining rotated rectangle around the body)
        # The 8 following corresponds to 4 (x,y) normalized keypoints coordinates (useful for determining rotated rectangle)
        # For full body, keypoints 1 and 2 are used.
        # For upper body, keypoints 3 and 4 are used.
        # Here we are only interested in full body

        # detections_to_rect (1st part)
        sqn_rr_center_xy = bbox[4:6]
        sqn_scale_xy = bbox[6:8]

        return torch.cat((score, sqn_rr_center_xy, sqn_scale_xy))

def test():

    model = DetectionBestCandidate()
    X = torch.randn(1, len(anchors), 1,  dtype=torch.float)
    Y = torch.randn(1, len(anchors), 12, dtype=torch.float)
    result = model(X, Y)
    print(result)

def export_onnx():
    """
    Exports the model to an ONNX file.
    """
    model = DetectionBestCandidate()
    X = torch.randn(1, len(anchors), 1,  dtype=torch.float)
    Y = torch.randn(1, len(anchors), 12, dtype=torch.float)
    onnx_name = "DetectionBestCandidate.onnx"

    print(f"Generating {onnx_name}")
    torch.onnx.export(
        model,
        (X, Y),
        onnx_name,
        opset_version=10,
        do_constant_folding=True,
        # verbose=True,
        input_names=['Identity_1', 'Identity'],
        output_names=['result']
    )

if __name__ == "__main__":

    test()
    export_onnx()