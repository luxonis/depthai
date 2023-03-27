from typing import Dict, List
import requests
import depthai as dai
from pathlib import Path
import json
from zipfile import ZipFile

ROBOFLOW_MODELS = Path.home() / Path('.cache/roboflow-models')

class RoboflowIntegration:
    def __init__(self, config: Dict):
        if 'key' not in config:
            raise ValueError("To download a model from Roboflow, you need to pass API key ('key')!")
        if 'model' not in config:
            raise ValueError("To download a model from Roboflow, you need to pass model path ('model')!")

        self.config = config

    def _file_with_ext(self, folder: Path, ext: str) -> Path:
        files = list(folder.glob(f"*{ext}"))
        if 0 == len(files):
            return None
        return files[0]

    def device_update(self, device: dai.Device) -> Path:
        mxid = device.getMxId()
        url = f"https://api.roboflow.com/depthai/{self.config['model']}/?api_key={self.config['key']}&device={mxid}"
        response = requests.get(url)

        name = self.config['model'].replace('/', '_') # '/' isn't valid folder name

        model_folder = ROBOFLOW_MODELS / name
        jsonFile = self._file_with_ext(model_folder, '.json')
        if jsonFile:
            return jsonFile

        json_res = response.json()
        if "error" in json_res:
            raise Exception(json_res['error'])

        response.raise_for_status()
        ret = json_res['depthai']

        if not str(ret['modelType']).startswith('yolov'):
            raise ValueError("This Roboflow's model is not from YOLO family!")

        if not str(ret['modelType']).endswith('n'):
            print('We recommend using a lighter version of the model to get a better performance!')

        print(f"Downloading '{ret['name']}' model from Roboflow server")

        zipFileReq = requests.get(ret['model'])
        zipFileReq.raise_for_status()

        (ROBOFLOW_MODELS / name).mkdir(parents=True, exist_ok=True)
        zipFilePath = str(ROBOFLOW_MODELS / 'roboflow.zip')
        # Download the .zip where our model is
        with open(zipFilePath, 'wb') as f:
            f.write(zipFileReq.content)

        print(f"Downloaded the model to {zipFilePath}")

        with ZipFile(zipFilePath, 'r') as zObject: # Extract the zip
            zObject.extractall(str(ROBOFLOW_MODELS / name))

        # Rename bin/xml files
        self._file_with_ext(model_folder, ".xml").rename(str(model_folder / (name + ".xml")))
        self._file_with_ext(model_folder, ".bin").rename(str(model_folder / (name + ".bin")))

        # Rename bin/xml paths inside the json
        new_json_name = str(model_folder / (name + ".json"))
        jsonConf = json.load(self._file_with_ext(model_folder, ".json").open())
        jsonConf['model']['xml'] = name + ".xml"
        jsonConf['model']['bin'] = name + ".bin"

        # For some reason, Roboflow server provides incorrect json file, so we have to edit it
        if 'output_format' in jsonConf:
            jsonConf['nn_config']['output_format'] = jsonConf['output_format']
            del jsonConf['output_format']
        if 'NN_family' in jsonConf:
            jsonConf['nn_config']['NN_family'] = jsonConf['NN_family']
            del jsonConf['NN_family']

        # Overwrite the json file, and rename it
        self._file_with_ext(model_folder, ".json").rename(new_json_name).open("w").write(json.dumps(jsonConf))

        Path(zipFilePath).unlink()  # Delete .zip

        return Path(new_json_name)
