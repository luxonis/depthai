from depthai_demo import DepthAI 
import threading
import time
import sys

if __name__ == "__main__":
    dai = DepthAI()
    passed = 1      # 1 is failed, 0 is passed

    thread = threading.Thread(target=dai.startLoop)
    thread.start()

    d2hLen = 0
    nnetLen = 0

    i=0
    while i<8:
        time.sleep(1)
        if dai.nnet_packets is not None:
            nnetLen += len(dai.nnet_packets)
        if dai.data_packets is not None:
            print("data packet in")
            for packet in dai.data_packets:
                print(packet.stream_name)
                if packet.stream_name == "meta_d2h":
                    d2hLen += 1
        i += 1

    if d2hLen > 0:
        passed = 0
    else:
        passed = 1
        
    dai.stopLoop()
    thread.join()

    sys.exit(passed)
