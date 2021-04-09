# Gen1 DepthAI API Demo Program

This repo contains demo application, which can load different networks, create pipelines, record video, etc.

__Documentation is available at [https://docs.luxonis.com](https://docs.luxonis.com).__

## Python modules (Dependencies)

DepthAI Demo requires [numpy](https://numpy.org/), [opencv-python](https://pypi.org/project/opencv-python/) and [depthai](https://github.com/luxonis/depthai-api). 
To get the versions of these packages you need for the program, use pip: (Make sure pip is upgraded: ` python3 -m pip install -U pip`)
```
python3 install_requirements.py
```

Optional:
For command line autocomplete when pressing TAB, only bash interpreter supported now:
Add to .bashrc:
`echo 'eval "$(register-python-argcomplete depthai_demo.py)"' >> ~/.bashrc`

If you use any other interpreter: https://kislyuk.github.io/argcomplete/

## Examples

`python3 depthai_demo.py` - depth & CNN inference example  

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

## OAK-D pass-through in KVM

OAK-D camera (possibly other Luxonis products) changes the USB device type when it is used by DepthAI API. <br>
This happens in backgound when the camera is used natively and the used does not see the change. But when the camera is used in a virtual environment the situation is different. If you are using virtual box then use the following [guide] (https://docs.luxonis.com/en/latest/pages/faq/#how-to-use-depthai-under-virtualbox). <br>
In order to have the camera accessible in the KVM virtual machine, there is the need to attach and detach USB devices on the fly when the host machine detects changes in the USB bus. For this reason there is the need for a udev rule which will help us creating the pass-through configurations. <br>

The following code is the contents of a udev rule which will call a script when OAK-D camera USB device is connected or disconnected from the USB bus. The file could be named: 80-movidius-host.rules. This file is installed in the host machine.
```
SUBSYSTEM=="usb", ACTION=="bind", ENV{ID_VENDOR_ID}=="03e7", MODE="0666", RUN+="/usr/local/bin/movidius_usb_hotplug.sh depthai-vm"
SUBSYSTEM=="usb", ACTION=="remove", ENV{PRODUCT}=="3e7/2485/1", ENV{DEVTYPE}=="usb_device", MODE="0666", RUN+="/usr/local/bin/movidius_usb_hotplug.sh depthai-vm"
SUBSYSTEM=="usb", ACTION=="remove", ENV{PRODUCT}=="3e7/f63b/100", ENV{DEVTYPE}=="usb_device", MODE="0666", RUN+="/usr/local/bin/movidius_usb_hotplug.sh depthai-vm"
```

The script that the udev rule is calling (movidius_usb_hotplug.sh) should then attach/detach the USB device to the virtual machine. In this case we need to call _virsh_ command. For example, the script could do the following:
```
#!/bin/bash

# Abort script execution on errors
set -e

if [ "${ACTION}" == 'bind' ]; then
  COMMAND='attach-device'
elif [ "${ACTION}" == 'remove' ]; then
  COMMAND='detach-device'
  if [ "${PRODUCT}" == '3e7/2485/1' ]; then
    ID_VENDOR_ID=03e7
    ID_MODEL_ID=2485
  fi
  if [ "${PRODUCT}" == '3e7/f63b/100' ]; then
    ID_VENDOR_ID=03e7
    ID_MODEL_ID=f63b
  fi
else
  echo "Invalid udev ACTION: ${ACTION}" >&2
  exit 1
fi

echo "Running virsh ${COMMAND} ${DOMAIN} for ${ID_VENDOR}." >&2
virsh "${COMMAND}" "${DOMAIN}" /dev/stdin <<END
<hostdev mode='subsystem' type='usb'>
  <source>
    <vendor id='0x${ID_VENDOR_ID}'/>
    <product id='0x${ID_MODEL_ID}'/>
  </source>
</hostdev>
END

exit 0
```
Note that when the device is disconnected from the USB bus, some udev environmental variables are not available (ID_VENDOR_ID or ID_MODEL_ID), that is why we need to use PRODUCT environmental variable to identify which device has been disconnected.
<br>
The virtual machine where DepthAI API application is running should have also defined a udev rules that identify the OAK-D camera. The udev rule is already decribed in the [Luxonis FAQ page](https://docs.luxonis.com/en/latest/pages/faq/).
