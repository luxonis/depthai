import RPi.GPIO as GPIO
import threading
import subprocess
import os
import time
import signal
import atexit
import argparse
from argparse import ArgumentParser


# Set up RPi GPIOs
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)   
GPIO.setup(12,GPIO.OUT)   # TST_PWR_ON
GPIO.setup(13,GPIO.IN)    # MOUNT_SENSE
GPIO.setup(26,GPIO.OUT)   # SYS_RST

# initialize gpios
GPIO.output(12,False)     # power off by default
GPIO.output(26,True)      # not in reset state by default


global p # To store child process info and id.

def cleanup():
    if(p is not None):
        print('Stopping subprocess with pid: ', str(p.pid))
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print('Stopped!')

def init():
    # Set up RPi GPIOs
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)   
    GPIO.setup(12,GPIO.OUT)   # TST_PWR_ON
    GPIO.setup(13,GPIO.IN)    # MOUNT_SENSE
    GPIO.setup(26,GPIO.OUT)   # SYS_RST

    # initialize gpios
    GPIO.output(12,False)     # power off by default
    GPIO.output(26,True)      # not in reset state by default
    return

def rst(sleeptime=0.1):
    GPIO.output(26,False)     # reset MX just in case
    time.sleep(sleeptime)
    GPIO.output(26,True)
    time.sleep(sleeptime)
    return

def pwron():
    #turns on power after checking module is present
    if moddetect()==True:
        print("Applying power to test board")
        GPIO.output(12,True)    # Turn power on to module
        time.sleep(1)    
        return(True)
    else:
        return(False)

def forcepoweron():
    print("Applying power to test board")
    #Turns power on without checking module is present
    GPIO.output(12,True)    # Turn power on to module
    time.sleep(1)    
    return   

def pwroff():
    print("Cutting power to test board")
    GPIO.output(12,False)    # Turn power on to module
    time.sleep(1)       
    return

def moddetect():
    if GPIO.input(13)==True: #Mounting module grounds GPIO, so logic is inverted. 
        return(False)
    else:
        return(True)

def rundepthai():
    global p
    test_cmd = """python3 depthai_demo.py -s left,10 right,10 previewout,10 metaout jpegout depth_raw,10"""
    p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setsid)
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
    isDetected = False
    printwaiting()
    while True:
        if moddetect() and not isDetected: # Starting child process if device detected and child process is not started.
            isDetected = True
            print("Module detected!")
            forcepoweron()
            rst()
            print("Starting test run...")
            rundepthai()
        elif not moddetect() and isDetected: # stoping child process if device is disconnected and child process is alive.
            isDetected = False
            print("Module unplugged!!!")
            print("Killing test run...")
            p.kill()
            pwroff()
            
            printwaiting()

            p = None


atexit.register(cleanup)

if __name__== "__main__":
  main()


