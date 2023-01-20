from marshmallow import Schema, fields

from depthai_sdk.classes.yolo_config import YoloConfig


class Mappings(Schema):
    # List of object detection labels. Labels can either be string, or array of [label, color] where color is in #RGB
    labels = fields.List(fields.Raw)


class Model(Schema):
    # A string, either path or url to .blob
    blob = fields.Str(required=False)
    # To download the model with blobconverter from either the open_model_zoo or the depthai-model-zoo
    model_name = fields.Str(required=False)
    zoo = fields.Str(required=False)
    # Model in IR format, uses blobconverter to convert the model to .blob.
    xml = fields.Str(required=False)
    bin = fields.Str(required=False)


class NNConfig(Schema):
    NN_family = fields.Str(required=False)
    confidence_threshold = fields.Float(required=False)
    NN_specific_metadata = fields.Nested(YoloConfig, required=False)
    output_format = fields.Str()
    input_size = fields.Str()


class Config(Schema):
    # Specifies where the model is/how to download it. It should contain one of the following
    model = fields.Nested(Model)
    # Path to the python script that contains decode() function that decodes dai.NNData into standardized NN results
    handler = fields.Str(required=False)
    nn_config = fields.Nested(NNConfig)  # Contains NN configuration data
    openvino_version = fields.Str(required=False)  # If the model requires specific OpenVINO version
    mappings = fields.Nested(Mappings, required=False)
    version = fields.Int()  # Version of the config file
