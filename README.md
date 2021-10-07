# DepthAI API Demo Program

This repo contains demo application, which can load different networks, create pipelines, record video, etc.

__Documentation is available at [https://docs.luxonis.com/](https://docs.luxonis.com/).__

## Python modules (Dependencies)

DepthAI Demo requires [numpy](https://numpy.org/), [opencv-python](https://pypi.org/project/opencv-python/) and [depthai](https://github.com/luxonis/depthai-python).
To get the versions of these packages you need for the program, use pip: (Make sure pip is upgraded: ` python3 -m pip install -U pip`)
```
python3 install_requirements.py
```

## Examples

`python3 depthai_demo.py` - RGB & CNN inference example

`python3 depthai_demo.py -vid <path_to_video_or_yt_link>` - CNN inference on video example

`python3 depthai_demo.py -cnn person-detection-retail-0013` - Run `person-detection-retail-0013` model from `resources/nn` directory

`python3 depthai_demo.py -cnn tiny-yolo-v3 -sh 8` - Run `tiny-yolo-v3` model from `resources/nn` directory and compile for 8 shaves

## Usage

```
$ depthai_demo.py --help

usage: depthai_demo.py [-h] [-cam {left,right,color}] [-vid VIDEO] [-dd] [-dnn] [-cnnp CNNPATH] [-cnn CNNMODEL] [-sh SHAVES] [-cnnsize CNNINPUTSIZE]
                       [-rgbr {1080,2160,3040}] [-rgbf RGBFPS] [-dct DISPARITYCONFIDENCETHRESHOLD] [-lrct LRCTHRESHOLD] [-sig SIGMA] [-med {0,3,5,7}] [-lrc] [-ext] [-sub]
                       [-dff] [-scale SCALE [SCALE ...]]
                       [-cm {AUTUMN,BONE,CIVIDIS,COOL,DEEPGREEN,HOT,HSV,INFERNO,JET,MAGMA,OCEAN,PARULA,PINK,PLASMA,RAINBOW,SPRING,SUMMER,TURBO,TWILIGHT,TWILIGHT_SHIFTED,VIRIDIS,WINTER}]
                       [-maxd MAXDEPTH] [-mind MINDEPTH] [-sbb] [-sbbsf SBBSCALEFACTOR]
                       [-s {nnInput,color,left,right,depth,depthRaw,disparity,disparityColor,rectifiedLeft,rectifiedRight} [{nnInput,color,left,right,depth,depthRaw,disparity,disparityColor,rectifiedLeft,rectifiedRight} ...]]
                       [--report {temp,cpu,memory} [{temp,cpu,memory} ...]] [--reportFile REPORTFILE] [-sync] [-monor {400,720,800}] [-monof MONOFPS] [-cb CALLBACK]
                       [--openvinoVersion {2020_3,2020_4,2021_1,2021_2,2021_3,2021_4}] [--count COUNTLABEL] [-dev DEVICEID] [-bandw {auto,low,high}] [-usbs {usb2,usb3}]
                       [-enc ENCODE [ENCODE ...]] [-encout ENCODEOUTPUT] [-xls XLINKCHUNKSIZE] [-camo CAMERAORIENTATION [CAMERAORIENTATION ...]] [--cameraControlls] 
                       [--cameraExposure CAMERAEXPOSURE] [--cameraSensitivity CAMERASENSITIVITY] [--cameraSaturation CAMERASATURATION] [--cameraContrast CAMERACONTRAST]
                       [--cameraBrightness CAMERABRIGHTNESS] [--cameraSharpness CAMERASHARPNESS]


optional arguments:
  -h, --help            show this help message and exit
  -cam {left,right,color}, --camera {left,right,color}
                        Use one of DepthAI cameras for inference (conflicts with -vid)
  -vid VIDEO, --video VIDEO
                        Path to video file (or YouTube link) to be used for inference (conflicts with -cam)
  -dd, --disableDepth   Disable depth information
  -dnn, --disableNeuralNetwork
                        Disable neural network inference
  -cnnp CNNPATH, --cnnPath CNNPATH
                        Path to cnn model directory to be run
  -cnn CNNMODEL, --cnnModel CNNMODEL
                        Cnn model to run on DepthAI
  -sh SHAVES, --shaves SHAVES
                        Number of MyriadX SHAVEs to use for neural network blob compilation
  -cnnsize CNNINPUTSIZE, --cnnInputSize CNNINPUTSIZE
                        Neural network input dimensions, in "WxH" format, e.g. "544x320"
  -rgbr {1080,2160,3040}, --rgbResolution {1080,2160,3040}
                        RGB cam res height: (1920x)1080, (3840x)2160 or (4056x)3040. Default: 1080
  -rgbf RGBFPS, --rgbFps RGBFPS
                        RGB cam fps: max 118.0 for H:1080, max 42.0 for H:2160. Default: 30.0
  -dct DISPARITYCONFIDENCETHRESHOLD, --disparityConfidenceThreshold DISPARITYCONFIDENCETHRESHOLD
                        Disparity confidence threshold, used for depth measurement. Default: 245
  -lrct LRCTHRESHOLD, --lrcThreshold LRCTHRESHOLD
                        Left right check threshold, used for depth measurement. Default: 4
  -sig SIGMA, --sigma SIGMA
                        Sigma value for Bilateral Filter applied on depth. Default: 0
  -med {0,3,5,7}, --stereoMedianSize {0,3,5,7}
                        Disparity / depth median filter kernel size (N x N) . 0 = filtering disabled. Default: 7
  -lrc, --stereoLrCheck
                        Enable stereo 'Left-Right check' feature.
  -ext, --extendedDisparity
                        Enable stereo 'Extended Disparity' feature.
  -sub, --subpixel      Enable stereo 'Subpixel' feature.
  -dff, --disableFullFovNn
                        Disable full RGB FOV for NN, keeping the nn aspect ratio
  -scale SCALE [SCALE ...], --scale SCALE [SCALE ...]
                        Define which preview windows to scale (grow/shrink). If scale_factor is not provided, it will default to 0.5 
                        Format: preview_name or preview_name,scale_factor 
                        Example: -scale color 
                        Example: -scale color,0.7 right,2 left,2
  -cm {AUTUMN,BONE,CIVIDIS,COOL,DEEPGREEN,HOT,HSV,INFERNO,JET,MAGMA,OCEAN,PARULA,PINK,PLASMA,RAINBOW,SPRING,SUMMER,TURBO,TWILIGHT,TWILIGHT_SHIFTED,VIRIDIS,WINTER}, --colorMap {AUTUMN,BONE,CIVIDIS,COOL,DEEPGREEN,HOT,HSV,INFERNO,JET,MAGMA,OCEAN,PARULA,PINK,PLASMA,RAINBOW,SPRING,SUMMER,TURBO,TWILIGHT,TWILIGHT_SHIFTED,VIRIDIS,WINTER}
                        Change color map used to apply colors to depth/disparity frames. Default: JET
  -maxd MAXDEPTH, --maxDepth MAXDEPTH
                        Maximum depth distance for spatial coordinates in mm. Default: 10000
  -mind MINDEPTH, --minDepth MINDEPTH
                        Minimum depth distance for spatial coordinates in mm. Default: 100
  -sbb, --spatialBoundingBox
                        Display spatial bounding box (ROI) when displaying spatial information. The Z coordinate get's calculated from the ROI (average)
  -sbbsf SBBSCALEFACTOR, --sbbScaleFactor SBBSCALEFACTOR
                        Spatial bounding box scale factor. Sometimes lower scale factor can give better depth (Z) result. Default: 0.3
  -s {nnInput,color,left,right,depth,depthRaw,disparity,disparityColor,rectifiedLeft,rectifiedRight} [{nnInput,color,left,right,depth,depthRaw,disparity,disparityColor,rectifiedLeft,rectifiedRight} ...], --show {nnInput,color,left,right,depth,depthRaw,disparity,disparityColor,rectifiedLeft,rectifiedRight} [{nnInput,color,left,right,depth,depthRaw,disparity,disparityColor,rectifiedLeft,rectifiedRight} ...]
                        Choose which previews to show. Default: []
  --report {temp,cpu,memory} [{temp,cpu,memory} ...]
                        Display device utilization data
  --reportFile REPORTFILE
                        Save report data to specified target file in CSV format
  -sync, --sync         Enable NN/camera synchronization. If enabled, camera source will be from the NN's passthrough attribute
  -monor {400,720,800}, --monoResolution {400,720,800}
                        Mono cam res height: (1280x)720, (1280x)800 or (640x)400. Default: 400
  -monof MONOFPS, --monoFps MONOFPS
                        Mono cam fps: max 60.0 for H:720 or H:800, max 120.0 for H:400. Default: 30.0
  -cb CALLBACK, --callback CALLBACK
                        Path to callbacks file to be used. Default: <project_root>/callbacks.py
  --openvinoVersion {2020_3,2020_4,2021_1,2021_2,2021_3,2021_4}
                        Specify which OpenVINO version to use in the pipeline
  --count COUNTLABEL    Count and display the number of specified objects on the frame. You can enter either the name of the object or its label id (number).
  -dev DEVICEID, --deviceId DEVICEID
                        DepthAI MX id of the device to connect to. Use the word 'list' to show all devices and exit.
  -bandw {auto,low,high}, --bandwidth {auto,low,high}
                        Force bandwidth mode. 
                        If set to "high", the output streams will stay uncompressed
                        If set to "low", the output streams will be MJPEG-encoded
                        If set to "auto" (default), the optimal bandwidth will be selected based on your connection type and speed
  -usbs {usb2,usb3}, --usbSpeed {usb2,usb3}
                        Force USB communication speed. Default: usb3
  -enc ENCODE [ENCODE ...], --encode ENCODE [ENCODE ...]
                        Define which cameras to encode (record) 
                        Format: cameraName or cameraName,encFps 
                        Example: -enc left color 
                        Example: -enc color right,10 left,10
  -encout ENCODEOUTPUT, --encodeOutput ENCODEOUTPUT
                        Path to directory where to store encoded files. Default: <project_root>
  -xls XLINKCHUNKSIZE, --xlinkChunkSize XLINKCHUNKSIZE
                        Specify XLink chunk size
  -camo CAMERAORIENTATION [CAMERAORIENTATION ...], --cameraOrientation CAMERAORIENTATION [CAMERAORIENTATION ...]
                        Define cameras orientation (available: AUTO, NORMAL, HORIZONTAL_MIRROR, VERTICAL_FLIP, ROTATE_180_DEG) 
                        Format: camera_name,camera_orientation 
                        Example: -camo color,ROTATE_180_DEG right,ROTATE_180_DEG left,ROTATE_180_DEG
  --cameraControlls      Show camera configuration options in GUI and controll them using keyboard
  --cameraExposure CAMERAEXPOSURE
                        Specify camera saturation
  --cameraSensitivity CAMERASENSITIVITY
                        Specify camera sensitivity
  --cameraSaturation CAMERASATURATION
                        Specify image saturation
  --cameraContrast CAMERACONTRAST
                        Specify image contrast
  --cameraBrightness CAMERABRIGHTNESS
                        Specify image brightness
  --cameraSharpness CAMERASHARPNESS
                        Specify image sharpness
```


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

       <path-to-openvino-folder>/deployment_tools/inference_engine/lib/intel64/myriad_compile -m ./ResNet50.xml -o ResNet50.blob -ip U8 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4

## Reporting issues

We are actively developing the DepthAI framework, and it's crucial for us to know what kind of problems you are facing.  
If you run into a problem, please follow the steps below and email support@luxonis.com:

1. Run `log_system_information.sh` and share the output from (`log_system_information.txt`).
2. Take a photo of a device you are using (or provide us a device model)
3. Describe the expected results;
4. Describe the actual running results (what you see after started your script with DepthAI)
5. How you are using the DepthAI python API (code snippet, for example)
6. Console output
