import sys
from pathlib import Path
import blobconverter
from difflib import get_close_matches


class BlobManager:
    zoo_name = None
    zoo_dir = None
    zoo_models = []
    use_zoo = False
    use_blob = False
    blob_path = None
    config_path = None

    def __init__(self, blob_path=None, config_path=None, zoo_name=None, zoo_dir=None):
        if blob_path is not None:
            self.blob_path = blob_path
            self.use_blob = True

        if zoo_dir is not None:
            self.zoo_dir = zoo_dir
            self.zoo_models = [f.stem for f in zoo_dir.iterdir() if f.is_dir()]
            self.use_zoo = True

        if zoo_name is not None:
            self.zoo_name = zoo_name
            self.use_zoo = True

        if config_path is not None:
            self.config_path = config_path


    def getBlob(self, shaves, openvino_version, target='auto'):
        version = openvino_version.name.replace("VERSION_", "").replace("_", ".")
        if self.use_blob:
            return self.blob_path
        elif self.use_zoo:
            try:
                self.blob_path = blobconverter.from_zoo(
                    name=self.zoo_name,
                    shaves=shaves,
                    version=version
                )
                self.use_blob = True
                return self.blob_path
            except Exception as e:
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if "not found in model zoo" in e.response.text:
                        all_models = set(self.zoo_models + blobconverter.zoo_list())
                        suggested = get_close_matches(self.zoo_name, all_models)
                        if len(suggested) > 0:
                            print("Model {} not found in model zoo. Did you mean: {} ?".format(self.zoo_name, " / ".join(suggested)), file=sys.stderr)
                        else:
                            print("Model {} not found in model zoo", file=sys.stderr)
                        raise SystemExit(1)
                    raise RuntimeError("Blob conversion failed with status {}! Error: \"{}\"".format(e.response.status_code, e.response.text))
                else:
                    raise
        elif self.config_path is not None:
            self.blob_path = blobconverter.compile_blob(
                version=version,
                blob_name=self.zoo_name,
                req_data={
                    "name": self.zoo_name,
                    "use_zoo": True,
                },
                req_files={
                    'config': self.config_path,
                },
                data_type="FP16",
                shaves=shaves,
            )
            self.use_blob = True
            return self.blob_path
