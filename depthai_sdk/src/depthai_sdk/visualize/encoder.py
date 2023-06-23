import dataclasses
import json

import numpy as np


class JSONEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)

        return json.JSONEncoder.default(self, obj)
