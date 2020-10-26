#!/usr/bin/env python3

#process watchdog, used to recover depthai-demo.py 
#on any userspace error in depthai-demo.py (segfault for example)
import os
import signal
import subprocess
import atexit
import sys

global p
global return_code

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
print(cmd)

atexit.register(cleanup)

run = True



while(run):
    p = subprocess.Popen(cmd, shell=True) 
    p.wait()
    return_code = p.returncode
    p=None
    print("Return code:"+str(return_code))
    if(return_code <= 4):
        run = False

exit(return_code)
