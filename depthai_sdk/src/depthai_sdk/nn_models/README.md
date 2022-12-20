# DepthAI-SDK supported NN models

We provide out-of-the-box support for many different models.

## Model list

- `mobilenet-ssd` - from [OMZ](https://docs.openvino.ai/latest/omz_models_model_mobilenet_ssd.html)

## Config file

All SDK supported model use config (`.json`) file which contains information on where the model is (or how to download it), it's configuration, output labels, logic on how to decode the results, etc.

#### Json structure

- `model`: Specifies where the model is/how to download it. It should contain one of the following:
  - `blob`: A string, either path or url to `.blob`
  - `model_name` (str) and `zoo_type` (str) to download the model with [blobconverter](https://github.com/luxonis/blobconverter) from either the [open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo) or the[depthai-model-zoo](https://github.com/luxonis/depthai-model-zoo)
  - `xml` (path/url) and `bin` (path/str) - model in IR format, uses blobconverter to convert the model to `.blob`.
- `handler`: Path to the python script that contains `decode()` function that decodes dai.NNData into standarized NN results ([see here](../classes/nn_results.py))
- `openvino_version`: OpenVINO version of the model
- `nn_config`: Contains NN configuration data;
  - `NN_family`: For on-device object detection decoding. Either `mobilenet` or `YOLO`.
  - `confidence_threshold`: Default object detection confidence to use.
  - `NN_specific_metadata`: Specific metadata for the NN, eg. YOLO requires anchors,masks,iou,etc.
- `mappings`:
  - `labels`: List of object detection labels. Labels can either be string, or array of `[label, color]` where color is in `#RGB`
- `version`: (int) version of the config file.

Example for `mobilenet-ssd`:
```
{
    "model":
    {
        "model_name": "mobilenet-ssd",
        "zoo": "intel"
    },
    "nn_config":
    {
        "NN_family" : "mobilenet",
        "confidence_threshold" : 0.5
    },
    "mappings":
    {
        "labels": ["background",  "aeroplane",  "bicycle",  "bird",  "boat",  "bottle",  "bus",  "car",  "cat",  "chair",  "cow",  "diningtable",  "dog",  "horse",  "motorbike",  "person",  "pottedplant",  "sheep",  "sofa",  "train",  "tvmonitor"]
    }
}
```

This will download the model from OMZ, and will use `(Spatial)MobileNetDetectionNetwork` node to decode object detection results on the device. Mappings (labels) are used for visualization of detected objects.