#!/usr/bin/env python3

import subprocess
import numpy as np
from pathlib import Path
import os, sys
import requests


def relative_to_abs_path(relative_path):
    dirname = Path(__file__).parent
    try:
        return (dirname / relative_path).resolve()
    except FileNotFoundError:
        return None


def download_model(model, model_zoo_folder, download_folder_path, openvino_version):

    model_downloader_path = relative_to_abs_path(f'openvino_{openvino_version}/downloader.py')
    
    model_downloader_options=f"--precisions FP16 --output_dir {download_folder_path} --cache_dir {download_folder_path}/.cache --num_attempts 2 --name {model} --model_root {model_zoo_folder}"
    model_downloader_options = model_downloader_options.split()
    downloader_cmd = [sys.executable, f"{model_downloader_path}"]
    downloader_cmd = np.concatenate((downloader_cmd, model_downloader_options))
    
    # print(downloader_cmd)
    result = subprocess.run(downloader_cmd)
    if result.returncode != 0:
        raise RuntimeError("Model downloader failed!")
    
    download_location = Path(download_folder_path) / model
    if(not download_location.exists()):
        raise RuntimeError(f"{download_location} doesn't exist for downloaded model!")


    return download_location



def convert_model_to_ir(model, model_zoo_folder, download_folder_path, openvino_version):

    converter_path = relative_to_abs_path(f'openvino_{openvino_version}/converter.py' )

    model_converter_options=f"--precisions FP16 --output_dir {download_folder_path} --download_dir {download_folder_path} --name {model} --model_root {model_zoo_folder}"
    model_converter_options = model_converter_options.split()
    converter_cmd = [sys.executable, f"{converter_path}"]
    converter_cmd = np.concatenate((converter_cmd, model_converter_options))
    # print(converter_cmd)
    result = subprocess.run(converter_cmd)
    if result.returncode != 0:
        raise RuntimeError("Model converter failed!")
    
    ir_model_location = Path(download_folder_path) / model / "FP16"


    return ir_model_location



def myriad_compile_model_local(shaves, xml_path, output_file, openvino_version):

    myriad_compile_path = None
    if myriad_compile_path is None:
        try:
            myriad_compile_path = Path(os.environ['INTEL_OPENVINO_DIR']) / 'deployment_tools/inference_engine/lib/intel64/myriad_compile'
        except KeyError:
            sys.exit('Unable to locate Model Optimizer. '
                + 'Use --mo or run setupvars.sh/setupvars.bat from the OpenVINO toolkit.')

    if openvino_version.startswith('2021'):
        config_file_content = {
            'MYRIAD_NUMBER_OF_SHAVES' : shaves,
            'MYRIAD_NUMBER_OF_CMX_SLICES' : shaves,
            'MYRIAD_THROUGHPUT_STREAMS' : 1
        }
    else:
        config_file_content = {
            'VPU_MYRIAD_PLATFORM' : 'VPU_MYRIAD_2480',
            'VPU_NUMBER_OF_SHAVES' : shaves,
            'VPU_NUMBER_OF_CMX_SLICES' : shaves,
            'VPU_MYRIAD_THROUGHPUT_STREAMS' : 1
        }

    parent_dir = Path(output_file).resolve().parents[0]
    config_file = parent_dir / 'myriad_compile_config.txt'
    
    print(f'Myriad config file {config_file}')
    with open(config_file, "w") as fp:
        for k, v in config_file_content.items():
            fp.write(str(k) + ' '+ str(v) + '\t\n')

    with open(config_file, "r") as fp:
        print(fp.read())

    myriad_compiler_options = f'-ip U8 -c {config_file} -m {xml_path} -o {output_file}'
    myriad_compiler_options = myriad_compiler_options.split()

    myriad_compile_cmd = np.concatenate(([myriad_compile_path], myriad_compiler_options))
    # print(myriad_compile_cmd)

    result = subprocess.run(myriad_compile_cmd)
    if result.returncode != 0:
        raise RuntimeError("Myriad compiler failed!")
    
    return output_file



def myriad_compile_model_cloud(xml, bin, shaves, output_file, openvino_version):
    PLATFORM="VPU_MYRIAD_2480"

    # use 69.214.171 instead luxonis.com to bypass cloudflare limitation of max file size
    url = "http://69.164.214.171:8083/compile"
    payload = {
        'compile_type': 'myriad',
        'compiler_params': f'-ip U8 -VPU_MYRIAD_PLATFORM {PLATFORM} -VPU_NUMBER_OF_SHAVES {shaves} -VPU_NUMBER_OF_CMX_SLICES {shaves}'
    }
    files = {
        'definition': open(Path(xml), 'rb'),
        'weights': open(Path(bin), 'rb')
    }
    params = {
        "version": openvino_version
    }
    try:
        response = requests.post(url, data=payload, files=files, params=params)
        response.raise_for_status()
    except Exception as ex:
        if getattr(ex, 'response', None) is None:
            print(f"Unknown error occured: {ex}")
            raise RuntimeError("Model compiler failed! Not connected to the internet?")
        print("Model compilation failed with error code: " + str(ex.response.status_code))
        print(str(ex.response.text))
        raise RuntimeError("Model compiler failed! Check logs for details")

    with open(output_file, "wb") as fp:
        fp.write(response.content)

    return output_file

def download_and_compile_NN_model(model, model_zoo_folder, shaves, nces, openvino_version, model_compilation_target='auto'):

    myriad_blob_path = relative_to_abs_path(f'openvino_{openvino_version}/myriad_blobs')
    output_location_dir = myriad_blob_path / model
    output_location_dir.mkdir(parents=True, exist_ok=True)

    output_file = model + ".sh" + str(shaves) + "NCE" + str(nces)
    output_location = output_location_dir / output_file

    #TODO improve caching based on the hash of yml file ?
    if(Path(output_location).exists()):
        print(f"Compiled mode found in cache: compiled for {shaves} shaves and {nces} NN_engines ")
        return output_location

    if model_compilation_target == 'auto' or model_compilation_target == 'local':
        try:
            openvino_dir = os.environ['INTEL_OPENVINO_DIR']
            print(f'Openvino installation detected {openvino_dir}') 

            installed_openvino_version_path = Path(openvino_dir) / "deployment_tools/model_optimizer/version.txt"
            installed_openvino_version = "not detected"
            with open(installed_openvino_version_path, "r") as fp:
                installed_openvino_version = fp.read()
                installed_openvino_version = installed_openvino_version.strip()
            print("Installed openvino version: ",installed_openvino_version)
            if openvino_version in installed_openvino_version:
                model_compilation_target = 'local'
                print(f'Supported openvino version installed: {openvino_version}')
            else:
                if model_compilation_target == 'local':
                    raise ValueError
                model_compilation_target = 'cloud'
                print(f'Unsupported openvino version installed at {openvino_dir}, version {installed_openvino_version}, supported version is: {openvino_version}')

        except:
            if model_compilation_target == 'local':
                raise SystemExit(f"Local model compilation was requested, but environment variables are not initialized for openvino {openvino_version}. Run setupvars.sh/setupvars.bat from the OpenVINO toolkit.")
            model_compilation_target = 'cloud'
    
    print(f'Model_compilation_target: {model_compilation_target}')
    print(f"Compiling model for {shaves} shaves, {nces} NN_engines ")


    download_folder_path  = relative_to_abs_path(f'openvino_{openvino_version}/downloads')
    download_location = download_model(model, model_zoo_folder, download_folder_path, openvino_version)

    
    if model_compilation_target == 'local':

        ir_model_location = convert_model_to_ir(model, model_zoo_folder, download_folder_path, openvino_version)

        if(not ir_model_location.exists()):
            raise RuntimeError(f"{ir_model_location} doesn't exist for downloaded model!")
        xml_path = ir_model_location / (model + ".xml")
        if(not xml_path.exists()):
            raise RuntimeError(f"{xml_path} doesn't exist for downloaded model!")
    
        return myriad_compile_model_local(shaves, xml_path, output_location, openvino_version)

    elif model_compilation_target == 'cloud':
         
        ir_model_location = Path(download_location) / "FP16"
        if(not ir_model_location.exists()):
            raise RuntimeError(f"{ir_model_location} doesn't exist for downloaded model!")
        xml_path = ir_model_location / (model + ".xml")
        if(not xml_path.exists()):
            raise RuntimeError(f"{xml_path} doesn't exist for downloaded model!")
        bin_path = ir_model_location / (model + ".bin")
        if(not bin_path.exists()):
            raise RuntimeError(f"{bin_path} doesn't exist for downloaded model!")

        return myriad_compile_model_cloud(xml=xml_path, bin=bin_path, shaves=shaves, output_file=output_location, openvino_version=openvino_version)

    else:
        assert 'model_compilation_target must be either : ["auto", "local", "cloud"]'

    return

def main(args):

    model = args['model_name']
    model_zoo_folder = args['model_zoo_folder']

    shaves = args['shaves']
    nces =  args['nces']
    model_compilation_target = args['model_compilation_target']
    openvino_version = args['openvino_version']

    return download_and_compile_NN_model(model, model_zoo_folder, shaves, nces, openvino_version, model_compilation_target)

if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser
    def parse_args():
        epilog_text = '''
        Myriad blob compiler.
        '''
        parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-model", "--model_name", default=None,
                            type=str, required=True,
                            help="model name")
        parser.add_argument("-sh", "--shaves", default=7, type=int, choices=range(1,17), metavar="[1,16]",
                            help="Number of shaves used by NN.")
        parser.add_argument("-nce", "--nces", default=1, type=int, choices=[1,2],
                            help="Number of NCEs used by NN.")
        parser.add_argument("-mct", "--model-compilation-target", default="auto",
                            type=str, required=False, choices=["auto","local","cloud"],
                            help="Compile model lcoally or in cloud?")
        parser.add_argument("-mz", "--model-zoo-folder", default=None,
                    type=str, required=True,
                    help="Path to folder with models")
        parser.add_argument("-op", "--openvino_version", default='2020.1',
                    type=str,
                    help="Openvino version for compilation")
        options = parser.parse_args()
        return options

    args = vars(parse_args())
    ret = main(args)
    print("Myriad blob written to: ",ret)

    exit(0)