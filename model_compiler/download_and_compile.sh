#!/bin/bash

OPENVINO_VERSION="2020.1.023"

echo_and_run() { echo -e "\$ $* \n" ; "$@" ; }

if [ $1 = "CLOUD_COMPILE" ] 
then
CLOUD_COMPILE="yes"
fi

CUR_DIR=$PWD
MODELS_LIST=`realpath models.txt`
DOWNLOADS_DIR=`realpath downloads`
NN_PATH=`realpath ../resources/nn`
OPENVINO_PATH=`realpath /opt/intel/openvino_$OPENVINO_VERSION`

if [ ! -d "$OPENVINO_PATH" ]; then
echo "OPENVINO_PATH doesn't exist! Openvino $OPENVINO_VERSION is not installed?"
exit 1
fi

if [ ! -d "$NN_PATH" ]; then
echo "NN_PATH doesn't exist"
exit 2
fi

MODEL_DOWNLOADER_OPTIONS="--precisions FP16 --output_dir $DOWNLOADS_DIR --cache_dir $DOWNLOADS_DIR --num_attempts 5 --list $MODELS_LIST"
MODEL_DOWNLOADER_PATH="$OPENVINO_PATH/deployment_tools/tools/model_downloader/downloader.py"


MYRIAD_COMPILE_OPTIONS="-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4"
MYRIAD_COMPILE_PATH="$OPENVINO_PATH/deployment_tools/inference_engine/lib/intel64/myriad_compile"

CLOUD_COMPILE_SCRIPT=$PWD/model_converter.py

echo $MODELS_LIST
rm -f $MODELS_LIST

cd $NN_PATH
for f in *; do
    if [ -d "$f" ]; then
        # $f is a directory
        echo "$f" >> $MODELS_LIST
    fi
done


echo_and_run python3 $MODEL_DOWNLOADER_PATH $MODEL_DOWNLOADER_OPTIONS

for f in *; do
    if [ -d "$f" ]; then
        # $f is a directory
        if [ "$CLOUD_COMPILE" = "yes" ] 
        then
            echo_and_run python3 $CLOUD_COMPILE_SCRIPT --xml $DOWNLOADS_DIR/intel/$f/FP16/$f.xml --bin $DOWNLOADS_DIR/intel/$f/FP16/$f.xml -o $f/$f.blob
        else
            echo_and_run $MYRIAD_COMPILE_PATH $MYRIAD_COMPILE_OPTIONS -m $DOWNLOADS_DIR/intel/$f/FP16/$f.xml -o $f/$f.blob
        fi
    fi
done

cd $CUR_DIR
