import importlib
import os
from pathlib import Path
from typing import Dict, Union

import requests

BLOBS_PATH = Path.home() / Path('.cache/blobs')


BLOBS_PATH = Path.home() / Path('.cache/blobs')

def getBlob(url: str) -> Path:
    """
    Download the blob from the URL and cache it locally.
    If the blob is already cached, return the cached file.
    
    @param url: URL to the blob
    @return: Local path to the blob
    """
    fileName = Path(url).name
    filePath = BLOBS_PATH / fileName

    print(filePath)
    
    # Check if the cached file exists and is a file
    if filePath.is_file():
        return filePath

    # Ensure the blobs cache directory exists
    BLOBS_PATH.mkdir(parents=True, exist_ok=True)

    # Download the blob
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Failed to download {url}. Status code: {r.status_code}")

    with open(filePath, 'wb') as f:
        f.write(r.content)
        print('Downloaded', fileName)

    return filePath


# Copied from utils.py - remove that once DepthAI Demo is deprecated
def loadModule(path: Path):
    """
    Loads module from specified path. Used internally e.g. to load a custom handler file from path

    Args:
        path (pathlib.Path): path to the module to be loaded

    Returns:
        module: loaded module from provided path
    """
    spec = importlib.util.spec_from_file_location(path.stem, str(path.absolute()))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Copid from utils.py (due to circular import)
def isUrl(source: Union[str, Path]) -> bool:
    if isinstance(source, Path):
        source = str(source)
    return source.startswith("http://") or source.startswith("https://")


def getSupportedModels(printModels=True) -> Dict[str, Path]:
    folder = Path(os.path.dirname(__file__)).parent / "nn_models"
    dic = dict()
    for item in folder.iterdir():
        if item.is_dir() and item.name != '__pycache__':
            dic[item.name] = item

    if printModels:
        print("\nDepthAI SDK supported models:\n")
        [print(f"- {name}") for name in dic]
        print('')
    return dic
