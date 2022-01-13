
import nmap
nmScan = nmap.PortScanner()
res = nmScan.scan(hosts='192.168.0.0/24', arguments='-sn')
# print(res['scan'])
for key in res['scan'].keys():
    print(res['scan'][key])
    print('----------------------------------------')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
res = nmScan.scan(hosts='192.168.0.0/24', arguments='-sP')
# print(res['scan'])
for key in res['scan'].keys():
    print(res['scan'][key])
    print('----------------------------------------')