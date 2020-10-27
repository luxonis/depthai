#!/bin/bash

set -e

python3 calibrate.py "$@"
python3 depthai_demo.py -s depth,12.0
