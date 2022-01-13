
import nmap
nmScan = nmap.PortScanner()
res = nmScan.scan(hosts='192.168.0.0/24', arguments='-sn')
print(res)

res = nmScan.scan(hosts='192.168.0.0/24', arguments='-sP')
print(res)