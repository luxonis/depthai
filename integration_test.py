import os
import signal
import subprocess
import time
import itertools

#todo merge with streams
stream_size = {
    "previewout": 300*300*3*30,
    "metaout" : 5 * 1024,
    "left"  : 1280*720*1*30,
    "right" : 1280*720*1*30,
    "depth_sipp" : 1280*720*2*30,
}

streams = [
    "previewout",
    "metaout",
    "left",
    "right",
    "depth_sipp"]

#todo ma
USB_version=3

if USB_version==2:
    gl_limit_fps=True
    usb_bandwith=29*1024*1024
    gl_max_fps=12.0
else:
    gl_limit_fps=False

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
            cmd = "python3 test.py " + config_builder
            print(cmd)
            pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                       shell=True, preexec_fn=os.setsid) 
                       
            time.sleep(5)
            #todo check return value
            os.killpg(os.getpgid(pro.pid), signal.SIGTERM)  # Send the signal to all the process groups
            time.sleep(5)
