FROM python:3.9-bullseye

RUN apt-get update && apt-get install -y wget build-essential cmake pkg-config libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran git

ADD docker_dependencies.sh .
RUN ./docker_dependencies.sh

ADD . /depthai

RUN python3 /depthai/install_requirements.py
