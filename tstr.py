import RPi.GPIO as GPIO
import os
import signal
import subprocess
import time
import itertools
import atexit
import sys
import argparse
from argparse import ArgumentParser


def parse_args():
    epilog_text = '''
    Test for the 1099 modules and 1098-based TLAs

    ADD EXAMPLE USAGE HERE
    '''
    parser = ArgumentParser(epilog=epilog_text,formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-tt", "--test_type", required=True, default=None, type=str,
                        help="Select '-tt module' for module test, and '-tt tla' for top-level assembly test.")

    options = parser.parse_args()
    print(options)
    return(options)

global args
try:
    args = vars(parse_args())
    print(args)
except:
    os._exit(2)

# Set up RPi GPIOs
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)   
GPIO.setup(12,GPIO.OUT)   # TST_PWR_ON
GPIO.setup(13,GPIO.IN)    # MOUNT_SENSE
GPIO.setup(26,GPIO.OUT)   # SYS_RST

# initialize gpios
GPIO.output(12,False)     # power off by default
GPIO.output(26,True)      # not in reset state by default


global all_processes
all_processes=list()
def cleanup():
    global run
    run=False
    timeout_sec = 5
    for p in all_processes: # list of your processes
        print("cleaning")
        p_sec = 0
        for _ in range(timeout_sec):
            if p.poll() == None:
                time.sleep(1)
                p_sec += 1
        if p_sec >= timeout_sec:
            p.killpg()
            # os.killpg(os.getpgid(p.pid), signal.SIGTERM)  # Send the signal to all the process groups
    print('cleaned up!')


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
        print("No module detected. Please mount module and retry.")
        return(False)
    else:
        print("Module detected.")
        return(True)
        
def rundepthai():
    args=""
    for arg in sys.argv[1:]:
        args+="'"+arg+"' "
    test_cmd = """python3 depthai.py"""

    atexit.register(cleanup)
    p = subprocess.run(test_cmd, shell=True)
    all_processes.append(p)
    #print("Testing for 10 seconds...")
    #time.sleep(10)
    return_code = p.returncode
    print("Return code:"+str(return_code))
    all_processes.clear()
    

def main():
    print(type(args['test_type']))
    if args['test_type'] is 'module':
        init() #initialize GPIOs
        while True:
            pwrsuccess = pwron()
            rst()
            if pwrsuccess:
                rundepthai()
    elif args['test_type'] is 'tla':
        forcepoweron()
        while True:
            rundepthai()
    else:
        print("Namespace not found")
            
        

if __name__== "__main__":
  main()



    
