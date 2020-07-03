#!/bin/bash
#example usage:
#cloud compile:
#./download_and_compile_all.sh 4 4 1 CLOUD_COMPILE
#local compile
#./download_and_compile_all.sh 4 4 1

RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(dirname "$0")"
cd $SCRIPT_DIR
SCRIPT_DIR=`realpath .`

echo_and_run() { echo -e "\$ $* \n" ; "$@" ; }

SHAVE_NR=$1
CMX_NR=$2
NCE_NR=$3

if [ "$NCE_NR" = "2" ]; then
cmx_odd=$(($CMX_NR%2))
shave_odd=$(($SHAVE_NR%2))
if [ "$cmx_odd" = "1" ]; then
    printf "${RED}CMX_NR config must be even number when NCE is 2!${NC}\n"
    exit 1
fi

if [ "$shave_odd" = "1" ]; then
    printf "${RED}SHAVE_NR config must be even number when NCE is 2!${NC}\n"
    exit 1
fi

fi


if [ "$4" = "CLOUD_COMPILE" ] 
then
CLOUD_COMPILE="CLOUD_COMPILE"
else
CLOUD_COMPILE=""
fi

NN_PATH=`realpath ../resources/nn`

cd $NN_PATH
for f in *; do
    if [ -z "$f" ]; then
        continue
    fi
    if [ -d "$f" ]; then
        echo_and_run $SCRIPT_DIR/download_and_compile.sh $f $SHAVE_NR $CMX_NR $NCE_NR $CLOUD_COMPILE
        if [ $? != 0 ]; then
            printf "${RED}Model compile failed with error: $? ${NC}\n"
        fi
    fi
done
