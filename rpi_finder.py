import os
import subprocess
from time import sleep, time
import argparse
import netifaces as ni
import paramiko
import argparse
import getpass

# response = os.system("nmap -sn  10.42.0.0/24 -oG - | awk '/Up$/{print $2}' ")
parser = argparse.ArgumentParser()
parser.add_argument("-interface", help="display a square of a given number", default="eno1", type=str)

args = parser.parse_args()


## TODO(sachin):Make the enp6s0 argument
host_ip = ni.ifaddresses(args.interface)[ni.AF_INET][0]['addr']
ip_val = host_ip.split('.')
print(ip_val)

net_msk = str()

net_msk = ip_val[0] + '.' + ip_val[1] + '.' + ip_val[2] + '.0/24'
uname = getpass.getuser()

print(net_msk)
ps = subprocess.Popen(('nmap', '-sP', net_msk), stdout=subprocess.PIPE)
output = subprocess.Popen(('awk', '/Up$/{print $2}'), stdin=ps.stdout, stdout=subprocess.PIPE)
stdout,stderr = ps.communicate()
nmap_response = stdout.decode("utf-8").splitlines()
print(nmap_response) 

ps.wait()

print("Printing response")
stdout,stderr = output.communicate()
op_response = stdout.decode("utf-8").splitlines()
print(op_response) 
# import nmap
# nmScan = nmap.PortScanner()
#  nmScan.scan(hosts='192.168.0.0/24', arguments='-n -sP -PE -PA21,23,80,3389')