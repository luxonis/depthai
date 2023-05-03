from marshmallow import Schema, fields


class YoloConfig(Schema):
    """
    Useful when parsing the YOLO config file from .json
    """
    classes = fields.Int()
    coordinates = fields.Int()
    anchors = fields.List(fields.Float)
    anchor_masks = fields.Dict(keys=fields.Str(), values=fields.List(fields.Int))
    iou_threshold = fields.Float()
    confidence_threshold = fields.Float()
