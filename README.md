# DepthAI Python API Extras

This repo contains the pre-compiled DepthAI Python module (compiled as an architecture-specific `.so` file), utilities, and DepthAI examples.

__Documentation is available at [https://docs.luxonis.com](https://docs.luxonis.com).__

## Python modules

Files with extention `.so` are python modules:  
- `depthai.cpython-36m-x86_64-linux-gnu.so` built for Ubuntu 18.04 & Python 3.6  
- `depthai.cpython-37m-arm-linux-gnueabihf.so` built for Raspbian 10 & Python 3.7  

## Examples

`test.py` - depth example  
`test_cnn.py` - CNN inference example

## Conversion of existing trained models into Intel Movidius binary format

OpenVINO toolkit contains components which allow conversion of existing supported trained `Caffe
and Tensorflow` models into Intel Movidius binary format through the Intermediate Representation
(IR) format.  
Example of the conversion:
1. The  command `<path-to-openvino-folder>/deployment_tools/model_optimizer/python3 mo.py --model_name ResNet50 --output_dir ResNet50_IR_FP16 --framework tf --data_type FP16 --input_model inference_graph.pb` will produce the following contents: `./ResNet50_IR_FP16, ResNet50.bin, ResNet50.mapping, ResNet50.xml.` Where `ResNet50.xml` is a file containing execution graph for this network; `ResNet50.bin` is weights file; and `ResNet50.mapping` contains mapping between layers in original public/custom model and layers within IR.
2. Weights (.bin) and graph (.xml) produced on the previous step will be required for building a blob file.
`myriad_compile` tool comes bundled with OpenVINO package and is used for blob generation. When producing blobs, the following constraints must be applied: `CMX-SLICES = 8 SHAVES = 8 INPUT-FORMATS = 8 OUTPUT-FORMATS= FP16/FP32 (Make changes in host code for meta frame display accordingly).` Example of command execution: `<path-to-openvino-folder>/deployment_tools/inference_engine/lib/intel64/myriad_compile -m ./ResNet50.xml -o ResNet50.blob -ip U8 -VPU_PLATFORM VPU_2480 -VPU_NUMBER_OF_SHAVES 8 -VPU_NUMBER_OF_CMX_SLICES 8`

## Reporting issues

We are activily developing the DepthAI framework, and it's crucial for us to know what kind of problems you are facing.  
If you run into a problem, please follow the steps below and email support@luxonis.com: 

1. Run `log_system_information.sh` and share the output from (`log_system_information.txt`).
2. Take a photo of a device you are using (or provide us a device model)
3. Describe the expected results; 
4. Describe the actual running results (what you see after started your script with DepthAI)
5. How you are using the DepthAI python API (code snippet, for example)
6. Console output
