import requests
import subprocess
import numpy as np
from pathlib import Path

from consts.resource_paths import nn_resource_path
from consts.resource_paths import model_downloader_path, download_folder_path
from consts.resource_paths import model_compiler_path

def download_model(model, shaves, cmx_slices, nces, output_file):

    model_downloader_options=f"--precisions FP16 --output_dir {download_folder_path} --cache_dir {download_folder_path}/.cache --num_attempts 5 --name {model} --model_root {nn_resource_path}"
    model_downloader_options = model_downloader_options.split()
    downloader_cmd = ["python3", f"{model_downloader_path}"]
    downloader_cmd = np.concatenate((downloader_cmd, model_downloader_options))
    # print(downloader_cmd)
    result = subprocess.run(downloader_cmd)
    if result.returncode != 0:
        raise RuntimeError("Model downloader failed! Not connected to the internet?")
    
    download_location = Path(download_folder_path) / model / "FP16"
    if(not download_location.exists()):
        raise RuntimeError(f"{download_location} doesn't exist for downloaded model!")
    xml_path = download_location / (model + ".xml")
    if(not xml_path.exists()):
        raise RuntimeError(f"{xml_path} doesn't exist for downloaded model!")
    bin_path = download_location / (model + ".bin")
    if(not bin_path.exists()):
        raise RuntimeError(f"{bin_path} doesn't exist for downloaded model!")


    output_location = Path(nn_resource_path) / model / output_file
    model_compiler_options=f"--xml {xml_path} --bin {bin_path} --output {output_location} --shaves {shaves} --cmx_slices {cmx_slices} --NCE {nces}"
    model_compiler_options = model_compiler_options.split()
    compiler_cmd = ["python3", f"{model_compiler_path}"]
    compiler_cmd = np.concatenate((compiler_cmd, model_compiler_options))
    # print(compiler_cmd)
    result = subprocess.run(compiler_cmd)
    if result.returncode != 0:
        raise RuntimeError("Model compiler failed! Not connected to the internet?")

    return 0