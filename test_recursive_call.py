# import RPi.GPIO as GPIO
import threading
import subprocess
import os
import time
import signal
import atexit
import argparse
from argparse import ArgumentParser
import psutil


global p # To store child process info and id.


def rundepthai():
    global p
    test_cmd = """python3 depthai_demo.py -s left,10 right,10 previewout,10 metaout jpegout depth,10 -monor 400"""
    p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setpgrp)
    return_code = p.returncode
    print("Return code:"+str(return_code))
    
def printwaiting():
    print() 
    print("---------------------")
    print("Waiting for module...")
    print("---------------------")
    print()

def main():
    global p
    # isDetected = False
    # printwaiting()
    while True:
        
        rundepthai()
            # p.kill()
        time.sleep(15)
        # os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        childs = psutil.Process(p.pid).children(recursive=True)
        print(childs)
        for pid in childs:
            os.kill(pid.pid, signal.SIGTERM)
        # p.kill()
        time.sleep(6)
        printwaiting()

        p = None




if __name__== "__main__":
  main()


