The default example runs a 20-class object detector (PASCAL VOC 20-class) with MobileNet-SSD 300x300:

20 classes:
Person: person
Animal: bird, cat, cow, dog, horse, sheep
Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

To run your own model, use OpenVINO to convert it as below:

*Inputs required*
BLOB file compatible with OpenVINO (Can be acquired through Exports function in inference-engine examples). They can also be generated using myriad_compile.
When producing blobs, the following constraints must be applied.
* CMX-SLICES    = 8
* SHAVES        = 8
* INPUT-FORMATS = 8
* OUTPUT-FORMATS= FP16/FP32

So to convert your model, open a terminal in the folder where the *.xml and *.bin files are stored and type:
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -m *.xml -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 8 -VPU_NUMBER_OF_CMX_SLICES 8

Example below is shown for mobilenet SSD:
/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/myriad_compile -m mobilenet-ssd-fp16.xml -ip U8 -VPU_PLATFORM VPU_2480 -VPU_NUMBER_OF_SHAVES 8 -VPU_NUMBER_OF_CMX_SLICES 8
