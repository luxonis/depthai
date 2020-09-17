#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import itertools
from threading import Timer
import atexit
import logging
import argparse
from argparse import ArgumentParser

def parse_args():
    epilog_text = '''
    Integration test for DepthAI.
    Generates all combinations of streams defined in "streams", runs each of them for maximum of "timeout" seconds.
    The logs are written into integration_test.log.
    
    Example usage: python3 integration_test.py -usb=2 -to=30
    python3 integration_test.py -usb=3 -to=60

    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-usb", "--usb_version", default=3,
                        type=int, required=False,
                        help="USB version on which to perform tests.")
    parser.add_argument("-to", "--timeout_sec", default=30, type=int,
                        help="Timeout in seconds for each stream combination. [MAX time allowed to run each test.]")
    options = parser.parse_args()

    return options

global args
try:
    args = vars(parse_args())
except:
    os._exit(2)

global USB_version
global timeout_sec

if args['usb_version']:
    USB_version = args['usb_version']
print("USB_version: "+str(USB_version))

if args['timeout_sec']:
    timeout_sec = args['timeout_sec']
print("timeout: "+str(timeout_sec) + " seconds")

logger = logging.getLogger('integration_test')
hdlr = logging.FileHandler('./integration_test.log', 'w')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def kill_proc(proc, timeout):
  timeout["value"] = True
  os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

def cleanup():
    global run
    run=False
    if(p is not None):
        print('Stopping subprocess with pid: ', str(p.pid))
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print('Stopped!')

#todo merge with streams
stream_size = {
    "previewout": 300*300*3*30,
    "metaout" : 5 * 1024,
    "left"  : 1280*720*1*30,
    "right" : 1280*720*1*30,
    "depth" : 1280*720*2*30,
}

streams = [
    "previewout",
    "metaout",
    "left",
    "right",
    "depth"]



global gl_limit_fps
if USB_version==2:
    gl_limit_fps=True
    usb_bandwith=29*1024*1024
    gl_max_fps=12.0
else:
    gl_limit_fps=False

global p
global return_code

p=None

atexit.register(cleanup)

config_ovewrite_cmd = """-co '{"streams": ["""
for L in range(0, len(streams)+1):
    for subset in itertools.combinations(streams, L):
        config_builder=config_ovewrite_cmd
        total_stream_throughput=0
        for comb1 in subset:
            total_stream_throughput+=stream_size[comb1]
        # print("total_stream_throughput",total_stream_throughput, "usb_bandwith", usb_bandwith)
        limit_fps=None
        if gl_limit_fps:
            if total_stream_throughput>usb_bandwith:
                limit_fps=True
            else:
                limit_fps=False
        else:
            limit_fps=False
        # print("limiting fps", limit_fps)
        for comb in subset:
            # print (subset.index(comb),comb, stream_size[comb])
            separator=None
            if subset.index(comb) is 0:
                separator=''
            else:
                separator=','
            if comb is not "metaout" and limit_fps:
                #todo dynamic max fps
                max_fps = gl_max_fps
                comb2 = '{"name": "' +comb+'", "max_fps": ' +str(max_fps)+ '}'
                config_builder = config_builder+separator+comb2
            else:
                separator+='"'
                config_builder = config_builder+separator+comb+'"'
        config_builder = config_builder + """]}'"""
        if subset:
            cmd = "python3 depthai_demo.py " + config_builder
            if(config_builder ==  """-co '{"streams": ["metaout"]}'"""):
                continue
            logger.info(cmd)

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
