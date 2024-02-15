import json
from pathlib import Path
from typing import Dict, Optional
from zipfile import ZipFile

import depthai as dai
from depthai_sdk.logger import LOGGER
import requests

ROBOFLOW_MODELS = Path.home() / Path('.cache/roboflow-models')


class RoboflowIntegration:
    def __init__(self, config: Dict):
        if 'key' not in config:
            raise ValueError("To download a model from Roboflow, you need to pass API key ('key')!")
        if 'model' not in config:
            raise ValueError("To download a model from Roboflow, you need to pass model path ('model')!")

        self.config = config

    def _file_with_ext(self, folder: Path, ext: str) -> Optional[Path]:
        files = list(folder.glob(f"*{ext}"))
        if 0 == len(files):
            return None
        return files[0]

    def device_update(self, device: dai.Device) -> Path:
        mxid = device.getMxId()
        name = self.config['model'].replace('/', '_')  # '/' isn't valid folder name

        model_folder = ROBOFLOW_MODELS / name
        json_file = self._file_with_ext(model_folder, '.json')
        if json_file:
            return json_file

        #Check the URL after checking the cache to make sure vision systems that are offline don't throw an exception
        url = f"https://api.roboflow.com/depthai/{self.config['model']}/?api_key={self.config['key']}&device={mxid}"
        response = requests.get(url)


        json_res = response.json()
        if "error" in json_res:
            raise Exception(json_res['error'])

        response.raise_for_status()
        ret = json_res['depthai']

        if not str(ret['modelType']).startswith('yolov'):
            raise ValueError("This Roboflow's model is not from YOLO family!")

        if not str(ret['modelType']).endswith('n'):
            LOGGER.info('We recommend using a lighter version of the model to get a better performance!')

        LOGGER.info(f"Downloading '{ret['name']}' model from Roboflow server")

        zip_file_req = requests.get(ret['model'])
        zip_file_req.raise_for_status()

        (ROBOFLOW_MODELS / name).mkdir(parents=True, exist_ok=True)
        zip_file_path = str(ROBOFLOW_MODELS / 'roboflow.zip')
        # Download the .zip where our model is
        with open(zip_file_path, 'wb') as f:
            f.write(zip_file_req.content)

        LOGGER.info(f"Downloaded the model to {zip_file_path}")

        with ZipFile(zip_file_path, 'r') as zObject:  # Extract the zip
            zObject.extractall(str(ROBOFLOW_MODELS / name))

        # Rename bin/xml files
        self._file_with_ext(model_folder, ".xml").rename(str(model_folder / (name + ".xml")))
        self._file_with_ext(model_folder, ".bin").rename(str(model_folder / (name + ".bin")))

        # Rename bin/xml paths inside the json
        new_json_name = str(model_folder / (name + ".json"))
        json_conf = json.load(self._file_with_ext(model_folder, ".json").open())
        json_conf['model']['xml'] = name + ".xml"
        json_conf['model']['bin'] = name + ".bin"

        # For some reason, Roboflow server provides incorrect json file, so we have to edit it
        if 'output_format' in json_conf:
            json_conf['nn_config']['output_format'] = json_conf['output_format']
            del json_conf['output_format']
        if 'NN_family' in json_conf:
            json_conf['nn_config']['NN_family'] = json_conf['NN_family']
            del json_conf['NN_family']

        # Overwrite the json file, and rename it
        self._file_with_ext(model_folder, ".json").rename(new_json_name).open("w").write(json.dumps(json_conf))

        Path(zip_file_path).unlink()  # Delete .zip
        return Path(new_json_name)
