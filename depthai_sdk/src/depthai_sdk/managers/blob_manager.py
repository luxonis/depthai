import sys
from pathlib import Path
import blobconverter
from difflib import get_close_matches
import depthai as dai


class BlobManager:
    """
    Manager class that handles MyriadX blobs.

    """

    _zoo_name = None
    _zoo_dir = None
    _blob_path = None
    _config_path = None
    _use_zoo = False
    _use_blob = False
    _zoo_models = []

    def __init__(self, blob_path:Path=None, config_path:Path=None, zoo_name:str=None, zoo_dir:Path=None):
        """
        Args:
            blob_path (pathlib.Path): Path to the compiled MyriadX blob file
            config_path (pathlib.Path): Path to model config file that is used to download the model
            zoo_name (str): Model name to be taken from model zoo
            zoo_dir (pathlib.Path): Path to model config file that is used to download the model
        """
        if blob_path is not None:
            self._blob_path = blob_path
            self._use_blob = True

        if zoo_dir is not None:
            self._zoo_dir = zoo_dir
            self._zoo_models = [f.stem for f in zoo_dir.iterdir() if f.is_dir()]
            self._use_zoo = True

        if zoo_name is not None:
            self._zoo_name = zoo_name
            self._use_zoo = True

        if config_path is not None:
            self._config_path = config_path


    def getBlob(self, shaves:int, openvino_version: dai.OpenVINO.Version):
        """
        This function is responsible for returning a ready to use MyriadX blob once requested.
        It will compile the model automatically using our online blobconverter tool. The compilation process will be
        ran only once, each subsequent call will return a path to previously compiled blob

        Args:
            shaves (int): Specify how many shaves the model will use. Range 1-16
            openvino_version (depthai.OpenVINO.Version): OpenVINO version which will be used to compile the MyriadX blob

        Returns:
            pathlib.Path: Path to compiled MyriadX blob

        Raises:
            SystemExit: If model name is not found in the zoo, this method will print all available onces and terminate
            RuntimeError: If conversion failed with unknown status
            Exception: If some unknown error will occur (reraise)
        """
        version = openvino_version.name.replace("VERSION_", "").replace("_", ".")
        if self._use_blob:
            return self._blob_path
        elif self._use_zoo:
            try:
                self._blob_path = blobconverter.from_zoo(
                    name=self._zoo_name,
                    shaves=shaves,
                    version=version
                )
                self._use_blob = True
                return self._blob_path
            except Exception as e:
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if "not found in model zoo" in e.response.text:
                        all_models = set(self._zoo_models + blobconverter.zoo_list())
                        suggested = get_close_matches(self._zoo_name, all_models)
                        if len(suggested) > 0:
                            print("Model {} not found in model zoo. Did you mean: {} ?".format(self._zoo_name, " / ".join(suggested)), file=sys.stderr)
                        else:
                            print("Model {} not found in model zoo", file=sys.stderr)
                        raise SystemExit(1)
                    raise RuntimeError("Blob conversion failed with status {}! Error: \"{}\"".format(e.response.status_code, e.response.text))
                else:
                    raise
        elif self._config_path is not None:
            name = self._config_path.parent.stem
            self._blob_path = blobconverter.compile_blob(
                version=version,
                blob_name=name,
                req_data={
                    "name": name,
                    "use_zoo": True,
                },
                req_files={
                    'config': self._config_path,
                },
                data_type="FP16",
                shaves=shaves,
            )
            self._use_blob = True
            return self._blob_path
