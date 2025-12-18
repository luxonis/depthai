#!/bin/bash

REPO_ROOT=$(git rev-parse --show-toplevel)
DOCKER_IMAGE=$(cat "$REPO_ROOT/ops/DOCKER_IMAGE")
xhost +local:root
docker run -it --rm \
    --privileged \
    --device /dev/bus/usb:/dev/bus/usb \
    --group-add lp \
    --security-opt apparmor=unconfined \
    --net=host \
    -h "docker_$(hostname)" \
    -e HOST_USER=$(whoami) \
    -e HOST_UID=$(id -u) \
    -e HOST_GID=$(id -g) \
    -e IN_PRODUCTION_TOOLS_DOCKER=1 \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /dev:/dev \
    -v $HOME/.luxonis_factory:/root/.luxonis_factory \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$(pwd)":/workspace \
    "$DOCKER_IMAGE"
