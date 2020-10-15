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
CMX_NR=$(($CMX_NR_OPT*2))
SHAVE_NR=$(($SHAVE_NR_OPT*2))
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


CLOUD_COMPILE_SCRIPT=$PWD/model_converter.py

DOWNLOADS_DIR=`realpath downloads`

if [ ! -d "$NN_PATH" ]; then
echo "NN_PATH doesn't exist"
exit 3
fi

MODELS_PATH="$CUR_DIR/../resources/nn"


MODEL_DOWNLOADER_OPTIONS="--precisions FP16 --output_dir $DOWNLOADS_DIR --cache_dir $DOWNLOADS_DIR --num_attempts 5 --name $MODEL_NAME --model_root $MODELS_PATH"
MODEL_DOWNLOADER_PATH="$CUR_DIR/downloader/downloader.py"

echo_and_run python3 $MODEL_DOWNLOADER_PATH $MODEL_DOWNLOADER_OPTIONS

BLOB_SUFFIX='.sh'$SHAVE_NR'cmx'$CMX_NR'NCE'$NCE_NR

BLOB_OUT=$NN_PATH/$MODEL_NAME/$MODEL_NAME.blob$BLOB_SUFFIX

if [ -f "$BLOB_OUT" ]; then
    echo "$MODEL_NAME.blob$BLOB_SUFFIX already compiled."
    exit 0
fi

cd $NN_PATH
if [ -d "$MODEL_NAME" ]; then
    echo_and_run python3 $CLOUD_COMPILE_SCRIPT --xml $DOWNLOADS_DIR/$MODEL_NAME/FP16/$MODEL_NAME.xml --bin $DOWNLOADS_DIR/$MODEL_NAME/FP16/$MODEL_NAME.bin -o $BLOB_OUT --shaves $SHAVE_NR_OPT --cmx_slices $CMX_NR_OPT --NCEs $NCE_NR
    if [ $? != 0 ]; then
        rm -f $BLOB_OUT
        exit 4
    fi
fi


cd $CUR_DIR
