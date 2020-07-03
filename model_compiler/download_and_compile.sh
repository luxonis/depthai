#!/bin/bash

#example usage:
#cloud compile:
#./download_and_compile.sh mobilenet-ssd 4 4 1 CLOUD_COMPILE
#local compile
#./download_and_compile.sh mobilenet-ssd 4 4 1


if [ "$#" -lt 4 ]; then
    echo "Invalid number of arguments."
    exit 1
fi

cd "$(dirname "$0")"

echo_and_run() { echo -e "\$ $* \n" ; "$@" ; }

MODEL_NAME=$1
SHAVE_NR=$2
CMX_NR=$3
NCE_NR=$4

if [ "$NCE_NR" = "0" ] 
then
VPU_MYRIAD_PLATFORM="VPU_MYRIAD_2450"
else
VPU_MYRIAD_PLATFORM="VPU_MYRIAD_2480"
fi

if [ "$NCE_NR" = "2" ]; then
CMX_NR_OPT=$(($CMX_NR/2))
SHAVE_NR_OPT=$(($SHAVE_NR/2))
else
CMX_NR_OPT=$CMX_NR
SHAVE_NR_OPT=$SHAVE_NR
fi


if [ "$5" = "CLOUD_COMPILE" ] 
then
CLOUD_COMPILE="yes"
fi

CUR_DIR=$PWD
NN_PATH=`realpath ../resources/nn`

if [ "$CLOUD_COMPILE" = "yes" ] 
then
CLOUD_COMPILE_SCRIPT=$PWD/model_converter.py
else

OPENVINO_VERSION="2020.1.023"

DOWNLOADS_DIR=`realpath downloads`
OPENVINO_PATH=`realpath /opt/intel/openvino_$OPENVINO_VERSION`

if [ ! -d "$OPENVINO_PATH" ]; then
echo "OPENVINO_PATH doesn't exist! Openvino $OPENVINO_VERSION is not installed?"
exit 2
fi

if [ ! -d "$NN_PATH" ]; then
echo "NN_PATH doesn't exist"
exit 3
fi

source $OPENVINO_PATH/bin/setupvars.sh

MODEL_DOWNLOADER_OPTIONS="--precisions FP16 --output_dir $DOWNLOADS_DIR --cache_dir $DOWNLOADS_DIR --num_attempts 5 --name $MODEL_NAME"
MODEL_DOWNLOADER_PATH="$OPENVINO_PATH/deployment_tools/tools/model_downloader/downloader.py"


MYRIAD_COMPILE_OPTIONS="-ip U8 -VPU_MYRIAD_PLATFORM $VPU_MYRIAD_PLATFORM -VPU_NUMBER_OF_SHAVES $SHAVE_NR_OPT -VPU_NUMBER_OF_CMX_SLICES $CMX_NR_OPT"
MYRIAD_COMPILE_PATH="$OPENVINO_PATH/deployment_tools/inference_engine/lib/intel64/myriad_compile"

echo_and_run python3 $MODEL_DOWNLOADER_PATH $MODEL_DOWNLOADER_OPTIONS

if [ -d "$DOWNLOADS_DIR/public" ]; then
cd $DOWNLOADS_DIR/public
if [ -d "$MODEL_NAME" ]; then
    # $MODEL_NAME is a directory
    mkdir -p $DOWNLOADS_DIR/intel/$MODEL_NAME/FP16/
    echo_and_run $OPENVINO_PATH/deployment_tools/model_optimizer/mo.py --input_model $DOWNLOADS_DIR/public/$MODEL_NAME/$MODEL_NAME.caffemodel --input_proto $DOWNLOADS_DIR/public/$MODEL_NAME/$MODEL_NAME.prototxt --data_type=FP16 --mean_values [127.5,127.5,127.5] --scale_values [255,255,255] -o $DOWNLOADS_DIR/intel/$MODEL_NAME/FP16/
fi
fi

fi

BLOB_SUFFIX='.sh'$SHAVE_NR'cmx'$CMX_NR'NCE'$NCE_NR

BLOB_OUT=$NN_PATH/$MODEL_NAME/$MODEL_NAME.blob$BLOB_SUFFIX

if [ -f "$BLOB_OUT" ]; then
    echo "$MODEL_NAME.blob$BLOB_SUFFIX already compiled."
    exit 0
fi

cd $NN_PATH
if [ -d "$MODEL_NAME" ]; then
    if [ "$CLOUD_COMPILE" = "yes" ] 
    then
        echo_and_run python3 $CLOUD_COMPILE_SCRIPT -i $MODEL_NAME -o $BLOB_OUT --shaves $SHAVE_NR_OPT --cmx_slices $CMX_NR_OPT --NCEs $NCE_NR
    else
        echo_and_run $MYRIAD_COMPILE_PATH $MYRIAD_COMPILE_OPTIONS -m $DOWNLOADS_DIR/intel/$MODEL_NAME/FP16/$MODEL_NAME.xml -o $BLOB_OUT
    fi
    if [ $? != 0 ]; then
        rm -f $BLOB_OUT
        exit 4
    fi
fi


cd $CUR_DIR
