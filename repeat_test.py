#!/usr/bin/env python3

import os
import signal
import subprocess
import time
from threading import Timer
import atexit
import logging
import sys


timeout_sec = 30


logger = logging.getLogger('repeat_test')
hdlr = logging.FileHandler('./repeat_test.log', 'w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def kill_proc(proc, timeout):
  timeout["value"] = True
  os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

global p
p=None

def cleanup():
    global run
    run=False
    if(p is not None):
        print('Stopping subprocess with pid: ', str(p.pid))
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print('Stopped!')

args=""

for arg in sys.argv[1:]:
    args+="'"+arg+"' "

cmd = "python3 depthai_demo.py " + args

logger.info(cmd)

atexit.register(cleanup)

while True:
    p = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    timeout = {"value": False}
    timer = Timer(timeout_sec, kill_proc, [p, timeout])
    timer.start()
    p.wait()
    timer.cancel()
    return_code = p.returncode
    p=None
    if(timeout["value"]):
        logger.info("returned succesfully")
    else:
        logger.info("returned with error code: " + str(return_code))
    time.sleep(5)
