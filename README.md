# DepthAI Python Module

This repo contains the pre-compiled DepthAI Python module (compiled as an architecture-specific `.so` file), utilities, and a submodule that allows compiling the DepthAI Python module for other platforms.

__Documentation is available at [https://docs.luxonis.com](https://docs.luxonis.com).__

## Python modules

DepthAI requires [numpy](https://numpy.org/) and [opencv-python](https://pypi.org/project/opencv-python/). To get the versions of these packages you need for DepthAI, use pip: `python3 -m pip install -r requirements.txt`

Optional:
For command line autocomplete when pressing TAB, only bash interpreter supported now:
Add to .bashrc:
`echo 'eval "$(register-python-argcomplete depthai.py)"' >> ~/.bashrc`

If you use any other interpreter: https://kislyuk.github.io/argcomplete/

Files with `.so` extension are the python modules:  
- `depthai.cpython-36m-x86_64-linux-gnu.so` built for Ubuntu 18.04 & Python 3.6  
- `depthai.cpython-37m-arm-linux-gnueabihf.so` built for Raspbian 10 & Python 3.7

For supporting other platforms, there is an option to build the python lib from sources by grabbing the [depthai-api](https://github.com/luxonis/depthai-api) submodule:

    git submodule update --init
    ./depthai-api/install_dependencies.sh # Only required in first build on a given system
    ./depthai-api/build_py_module.sh

When updating DepthAI on these platforms it is often necessary to run `./depthai-api/build_py_module.sh --clean` in order to build a new version of the depthai-api module for your chosen platform. 

## Examples

`python3 test.py` - depth & CNN inference example  

## Conversion of existing trained models into Intel Movidius binary format

OpenVINO toolkit contains components which allow conversion of existing supported trained `Caffe` and `Tensorflow` models into Intel Movidius binary format through the Intermediate Representation (IR) format.

Example of the conversion:
1. First the `model_optimizer` tool will convert the model to IR format:  

       cd <path-to-openvino-folder>/deployment_tools/model_optimizer
       python3 mo.py --model_name ResNet50 --output_dir ResNet50_IR_FP16 --framework tf --data_type FP16 --input_model inference_graph.pb

    - The command will produce the following files in the `ResNet50_IR_FP16` directory:
        - `ResNet50.bin` - weights file;
        - `ResNet50.xml` - execution graph for the network;
        - `ResNet50.mapping` - mapping between layers in original public/custom model and layers within IR.
2. The weights (`.bin`) and graph (`.xml`) files produced above (or from the Intel Model Zoo) will be required for building a blob file,
with the help of the `myriad_compile` tool. When producing blobs, the following constraints must be applied:

       CMX-SLICES = 4 
       SHAVES = 4 
       INPUT-FORMATS = 8 
       OUTPUT-FORMATS = FP16/FP32 (host code for meta frame display should be updated accordingly)

    Example of command execution:

       <path-to-openvino-folder>/deployment_tools/inference_engine/lib/intel64/myriad_compile -m ./ResNet50.xml -o ResNet50.blob -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4

## Reporting issues

We are actively developing the DepthAI framework, and it's crucial for us to know what kind of problems you are facing.  
If you run into a problem, please follow the steps below and email support@luxonis.com: 

1. Run `log_system_information.sh` and share the output from (`log_system_information.txt`).
2. Take a photo of a device you are using (or provide us a device model)
3. Describe the expected results; 
4. Describe the actual running results (what you see after started your script with DepthAI)
5. How you are using the DepthAI python API (code snippet, for example)
6. Console output
