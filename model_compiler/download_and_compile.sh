#!/bin/bash
cd "$(dirname "$0")"

echo_and_run() { echo -e "\$ $* \n" ; "$@" ; }

MODEL_NAME=$1
SHAVE_NR=$2
CMX_NR=$3
NCE_NR=$4

CLOUD_COMPILE="yes"

CUR_DIR=$PWD
NN_PATH=`realpath ../resources/nn`

BLOB_SUFFIX='.sh'$SHAVE_NR'cmx'$CMX_NR
if [ $NCE_NR == 0 ]; then
    BLOB_SUFFIX=$BLOB_SUFFIX"NO_NCE"
fi

BLOB_OUT=$NN_PATH/$MODEL_NAME/$MODEL_NAME.blob$BLOB_SUFFIX

if [ -f "$BLOB_OUT" ]; then
    echo "$MODEL_NAME.blob$BLOB_SUFFIX already compiled."
    exit 0
fi

CLOUD_COMPILE_SCRIPT=$PWD/model_converter.py

cd $NN_PATH
if [ -d "$MODEL_NAME" ]; then
    echo_and_run python3 $CLOUD_COMPILE_SCRIPT -i $MODEL_NAME -o $BLOB_OUT --shaves $SHAVE_NR --cmx_slices $CMX_NR --NCEs $NCE_NR
    if [ $? != 0 ]; then
        rm -f $BLOB_OUT
        exit 2
    fi
fi


cd $CUR_DIR
