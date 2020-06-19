#!/bin/bash
cd "$(dirname "$0")"

OPENVINO_VERSION="2020.1.023"

echo_and_run() { echo -e "\$ $* \n" ; "$@" ; }

MODEL_NAME=$1
SHAVE_NR=$2
CMX_NR=$3

CLOUD_COMPILE="yes"

CUR_DIR=$PWD
DOWNLOADS_DIR=`realpath downloads`
NN_PATH=`realpath ../resources/nn`

MODEL_DOWNLOADER_OPTIONS="--precisions FP16 --output_dir $DOWNLOADS_DIR --cache_dir $DOWNLOADS_DIR --num_attempts 5 --name $MODEL_NAME"
MODEL_DOWNLOADER_PATH=`realpath open_model_zoo/tools/downloader/downloader.py`

BLOB_SUFFIX='.sh'$SHAVE_NR'cmx'$CMX_NR

BLOB_OUT=$NN_PATH/$MODEL_NAME/$MODEL_NAME.blob$BLOB_SUFFIX

if [ -f "$BLOB_OUT" ]; then
    echo "$MODEL_NAME.blob$BLOB_SUFFIX already compiled."
    exit 0
fi


CLOUD_COMPILE_SCRIPT=$PWD/model_converter.py



cd $NN_PATH
if [ -d "$MODEL_NAME" ]; then
    echo_and_run python3 $CLOUD_COMPILE_SCRIPT -i $MODEL_NAME -o $BLOB_OUT --shaves $SHAVE_NR --cmx_slices $CMX_NR 
    if [ $? != 0 ]; then
        rm -f $BLOB_OUT
        exit 2
    fi
fi


cd $CUR_DIR
