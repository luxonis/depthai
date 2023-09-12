# DepthAI API Demo Program

[![Discord](https://img.shields.io/discord/790680891252932659?label=Discord)](https://discord.gg/luxonis)
[![Forum](https://img.shields.io/badge/Forum-discuss-orange)](https://discuss.luxonis.com/)
[![Docs](https://img.shields.io/badge/Docs-DepthAI-yellow)](https://docs.luxonis.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repo contains demo application, which can load different networks, create pipelines, record video, etc.

_Click on the GIF below to see a full example run_

[![depthai demo](https://user-images.githubusercontent.com/5244214/142426845-82f5f8fd-ad1a-4873-97a5-2b3fcdb0ca2e.gif)](https://www.youtube.com/watch?v=sCZpsFQBffk)


__Documentation is available at [https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#first-steps-with-depthai).__

## Installation
First you need to clone this repository with
```
git clone --recursive https://github.com/luxonis/depthai.git
```
In case you have repository already cloned, you can update your submodules with:
```
git pull --recurse-submodules 
```
There are two installation steps that need to be performed to make sure the demo works:
- **One-time installation** that will add all necessary packages to your OS.
  ```
  $ sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
  ```
  Please follow [this installation page](https://docs.luxonis.com/projects/api/en/latest/install/) to see instructions for other platforms


- **Python dependencies installation** that makes sure your Python interpreter has all required packages installed.
  This script is safe to be run multiple times and should be ran after every demo update
  ```
  $ python3 install_requirements.py
  ```

## Docker

One may start any DepthAI programs also through Docker:
(Allowing X11 access from container might be required: `xhost local:root`)

DepthAI Demo
```
docker run --privileged -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network host --rm -i -t luxonis/depthai:latest python3 /depthai/depthai_demo.py
```

Calibration
```
docker run --privileged -v /dev/bus/usb:/dev/bus/usb --device-cgroup-rule='c 189:* rmw' -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network host --rm -i -t luxonis/depthai:latest python3 /depthai/calibrate.py [parameters]
```


## Usage

This repository and the demo script itself can be used in various independent cases:
- As a tool to try out different DepthAI features, explorable either with command line arguments (with `--guiType cv`) or in clickable QT interface (with `--guiType qt`)
- As a quick prototyping backbone, either utilising [callbacks mechanism](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#callbacks-file) or by [extending the `Demo` class itself](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#importing-demo-as-a-class)

### QT GUI usage

See instuctions [here](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#default-run)

![qt demo](https://user-images.githubusercontent.com/5244214/142722740-47e545b7-c7fe-4132-9704-ae3b47d60957.png)

### Command line usage

**Examples**

`python3 depthai_demo.py -gt cv` - RGB & CNN inference example

`python3 depthai_demo.py -gt cv -vid <path_to_video_or_yt_link>` - CNN inference on video example

`python3 depthai_demo.py -gt cv -cnn person-detection-retail-0013` - Runs `person-detection-retail-0013` model from `resources/nn` directory

`python3 depthai_demo.py -gt cv -cnn tiny-yolo-v3 -sh 8` - Run `tiny-yolo-v3` model from `resources/nn` directory and compile for 8 shaves

**Demo**

![cv demo](https://user-images.githubusercontent.com/18037362/177605180-8f327513-131d-4c31-9829-3579bd717c6a.jpg)

For the full reference, run ``$ depthai_demo.py --help``.

## DepthAI Apps

We currently have 2 apps, ``uvc`` and ``record``.

### UVC app

This app will upload an UVC pipeline to the connected OAK camera, so you will be able to use an [OAK as a webcam](https://docs.luxonis.com/en/latest/pages/oak_webcam/).

### Record App

Record app lets you record encoded and synced video streams (MJPEG/H265) across all available devices into .mp4, [Foxglove's .MCAP](https://mcap.dev/), or [ROS' .bag ](http://wiki.ros.org/rosbag). Since mono streams are synced, you will be able to reconstruct the whole stereo depth perception.

Run using ``$ depthai_demo.py -app record [-p SAVE_PATH] [-q QUALITY] [--save STREAMS] [-fc] [-tl]``. More information about the arguments and replaying can be [found here](https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay).

## Supported models

We have added support for a number of different AI models that work (decoding and visualization) out-of-the-box with the demo. You can specify which model to run with `-cnn` argument, as shown above. Currently supported models:

```
- deeplabv3p_person
- face-detection-adas-0001
- face-detection-retail-0004
- human-pose-estimation-0001
- mobilenet-ssd
- openpose2
- pedestrian-detection-adas-0002
- person-detection-retail-0013
- person-vehicle-bike-detection-crossroad-1016
- road-segmentation-adas-0001
- tiny-yolo-v3
- vehicle-detection-adas-0002
- vehicle-license-plate-detection-barrier-0106
- yolo-v3
```

If you would like to use a custom AI model, see [documentation here](https://docs.luxonis.com/en/latest/pages/tutorials/first_steps/#using-custom-models).

## Usage statistics

By default, the demo script will collect anonymous usage statistics during runtime. These include:
- Device-specific information (like mxid, connected cameras, device state and connection type)
- Environment-specific information (like OS type, python version, package versions)

We gather this data to better understand what environemnts are our users using, as well as assist better in support questions.

**All of the data we collect is anonymous and you can disable it at any time**. To do so, click on the "Misc" tab and disable sending the statistics or create a `.consent` file in repository root with the following content

```
{"statistics": false}
```

## Reporting issues

We are actively developing the DepthAI framework, and it's crucial for us to know what kind of problems you are facing.
If you run into a problem, please follow the steps below and email support@luxonis.com:

1. Run `log_system_information.sh` and share the output from (`log_system_information.txt`).
2. Take a photo of a device you are using (or provide us a device model)
3. Describe the expected results;
4. Describe the actual running results (what you see after started your script with DepthAI)
5. How you are using the DepthAI python API (code snippet, for example)
6. Console output
