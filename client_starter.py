import os
import subprocess
from time import sleep, time
import argparse
import netifaces as ni
import paramiko
import argparse

# response = os.system("nmap -sn  10.42.0.0/24 -oG - | awk '/Up$/{print $2}' ")
parser = argparse.ArgumentParser()
parser.add_argument("-fusb2", "--force_usb2", default=False, action="store_true",
                            help="Force usb2 connection")
parser.add_argument("-interface", help="display a square of a given number", default="enp6s0", type=str)

args = parser.parse_args()


## TODO(sachin):Make the enp6s0 argument
host_ip = ni.ifaddresses(args.interface)[ni.AF_INET][0]['addr']
ip_val = host_ip.split('.')
print(ip_val)

net_msk = str()

net_msk = ip_val[0] + '.' + ip_val[1] + '.' +ip_val[2] + '.0/24' 

print(net_msk)
localpath = '/home/sachin/Desktop/luxonis/depthai/calibration_client.py'
remotepath = '/home/pi/calibration_client.py'


ps = subprocess.Popen(('nmap', '-sn', net_msk, '-oG', '-'), stdout=subprocess.PIPE)
output = subprocess.Popen(('awk', '/Up$/{print $2}'), stdin=ps.stdout, stdout=subprocess.PIPE)
ps.wait()

print("Printing response")
stdout,stderr = output.communicate()
op_response = stdout.decode("utf-8").splitlines()
print(op_response)

# rpi_ip = 'pi@' + op_response[1] 

# print("Printing ip")
# print(op_response[1])

ip = None
for addr in op_response:
    if addr != host_ip:
        ip = addr

print("prinitng selected address client ")
print(ip)

ssh = paramiko.SSHClient()
# ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print("creating ssd client")

ssh.connect(ip, username='pi', password='luxonis')

sftp = ssh.open_sftp()
sftp.put(localpath, remotepath)
sftp.close()
print("executing on clinet")

stdin, stdout, stderr = ssh.exec_command('python3 calibration_client.py')
# stdin, stdout, stderr = ssh.exec_command('ls')

str_op = None
str_op = stdout.read()
while str_op:
    print(str_op.decode("utf-8"))
    str_op = stdout.read()

str_op = None
str_op = stderr.read()
while str_op:
    print(str_op.decode("utf-8"))
    str_op = stderr.read()

print("prinitng on clinet")
ssh.close()




''' ~~ Test script ~~
import paramiko
from time import sleep 

localpath = '/home/sachin/Desktop/luxonis/depthai/calibration_client.py'
remotepath = '/home/pi/calibration_client.py'

ssh = paramiko.SSHClient()
# ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect('10.42.0.56', username='pi', password='luxonis')
print("creating ssd client")

# sftp = ssh.open_sftp()
# sftp.put(localpath, remotepath)
# sftp.close()
# channel = ssh.invoke_shell()
print("channel created")

stdin, stdout, stderr = ssh.exec_command('ls')
print("prinitng on clinet")

print(stdout.read().decode("utf-8"))
sleep(3)
ssh.close()
'''