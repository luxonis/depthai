FROM luxonis/depthai-library:ubuntu-develop

ENV PYTHONPATH ""

RUN git clone https://github.com/luxonis/depthai /depthai

WORKDIR /depthai
RUN python3 install_requirements.py

CMD ["python3", "depthai_demo.py", "--skipUdevCheck"]