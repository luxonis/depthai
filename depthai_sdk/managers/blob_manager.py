import sys
from pathlib import Path
import blobconverter
from difflib import get_close_matches


class BlobManager:
    def __init__(self, model_name=None, model_dir:Path=None):
        self.model_dir = None
        self.zoo_dir = None
        self.config_file = None
        self.blob_path = None
        self.use_zoo = False
        self.use_blob = False
        self.zoo_models = [f.stem for f in model_dir.parent.iterdir() if f.is_dir()] if model_dir is not None else []
        if model_dir is None:
            self.model_name = model_name
            self.use_zoo = True
        else:
            self.model_dir = Path(model_dir)
            self.zoo_dir = self.model_dir.parent
            self.model_name = model_name or self.model_dir.name
            self.config_file = self.model_dir / "model.yml"
            blob = next(self.model_dir.glob("*.blob"), None)
            if blob is not None:
                self.use_blob = True
                self.blob_path = blob
            if not self.config_file.exists():
                self.use_zoo = True

    def compile(self, shaves, openvino_version, target='auto'):
        version = openvino_version.name.replace("VERSION_", "").replace("_", ".")
        if self.use_blob:
            return self.blob_path
        elif self.use_zoo:
            try:
                self.blob_path = blobconverter.from_zoo(
                    name=self.model_name,
                    shaves=shaves,
                    version=version
                )
                return self.blob_path
            except Exception as e:
                if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                    if "not found in model zoo" in e.response.text:
                        all_models = set(self.zoo_models + blobconverter.zoo_list())
                        suggested = get_close_matches(self.model_name, all_models)
                        if len(suggested) > 0:
                            print("Model {} not found in model zoo. Did you mean: {} ?".format(self.model_name, " / ".join(suggested)), file=sys.stderr)
                        else:
                            print("Model {} not found in model zoo", file=sys.stderr)
                        raise SystemExit(1)
                    raise RuntimeError("Blob conversion failed with status {}! Error: \"{}\"".format(e.response.status_code, e.response.text))
                else:
                    raise
        else:
            self.blob_path = blobconverter.compile_blob(
                version=version,
                blob_name=self.model_name,
                req_data={
                    "name": self.model_name,
                    "use_zoo": True,
                },
                req_files={
                    'config': self.config_file,
                },
                data_type="FP16",
                shaves=shaves,
            )
            return self.blob_path
