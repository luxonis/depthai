# DepthAI Python API Extras

This repo contains the pre-compiled DepthAI Python module (compiled as an architecture-specific `.so` file), utilities, and DepthAI examples.

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


## Calibration

For better depth image quality, perform a stereo camera calibration. Follow these steps:

1. Print the chessboard calibration image. This image is in the `resources` folder (resources/calibration-chess-board.png). Measure the square size in centimeters and insert the value in the command below. The entire board should fit on a single piece of paper (scale to fit).
2. Start the calibration script: `python3 calibrate.py -s [SQUARE_SIZE_IN_CM]`. Left and right video streams are displayed. Each window will contain a polygon.  
3. Hold up the printed chessboard so that the whole of the checkerboard is displayed within both video streams. Match the orientation of the overlayed polygon and press [SPACEBAR] to capture an image. The checkerboard pattern does not need to match the polygon exactly, but it is important to use the polygon as a guideline for angling and location relative to the camera. There are 13 required polygon positions.
4. After capturing images for all of the polygon positions, the calibration image processing step will begin. If successful, a `depthai.calib` file will be added to the `resources/` folder. This file is loaded by default via the `calib_fpath` variable within `consts/resource_paths.py`. 

## Issues reporting  

We are developing depthai framework, and it's crucial for us to know what kind of problems users are facing.  
So thanks for testing DepthAI! The information you give us, the faster we will help you and make depthai better!  

Please, do the following steps:  
1. Run script `log_system_information.sh` and provide us the output (`log_system_information.txt`, it's system version & modules versions);  
2. Take a photo of a device you are using (or provide us a device model);  
3. Describe the expected results;  
4. Describe the actual running results (what you see after started your script with depthai);  
5. Provide us information about how you are using the depthai python API (code snippet, for example);  
6. Send us console outputs;  
