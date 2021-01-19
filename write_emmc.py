# import blkinfo
from blkinfo.filters import BlkDiskInfo
import json
import os
import re
import subprocess
from time import sleep 
# device_re = re.compile("Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$", re.I)
# df = subprocess.check_output("lsusb")
# devices = []
# for i in df.split('\n'):
#     if i:
#         info = device_re.match(i)
#         if info:
#             dinfo = info.groupdict()
#             dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
#             devices.append(dinfo)
# print(devices)


# install rpiboot first
def start_rpi_boot():
    response = os.system("sudo ./rpiboot ")
    # and then check the response...
    if response == 0:
        eMMC_status = True
    else:
        eMMC_status = False

    return eMMC_status


start_rpi_boot()

myblkd = BlkDiskInfo()
filters = {
        'name': "sd*"
    }
all_my_disks = myblkd.get_disks(filters)
json_output = json.dumps(all_my_disks)
print(json_output)

response_status = []

for item in json_output:
    test_cmd = "sudo dd if=DepthAI_uSD_AS_RPI_202004161009.img of=/dev/" + item["name"] + " bs=4M conv=fsync status=progress"
    p = subprocess.Popen(test_cmd, shell=True, preexec_fn=os.setsid)
    response_status.append(p)

is_finished = False 
while not is_finished:
    temp_status = True
    for proc in response_status:
        if proc.poll() is None:
            temp_status = False
    sleep(1)
    is_finished = temp_status

print("Finishing writing to eMMC")








