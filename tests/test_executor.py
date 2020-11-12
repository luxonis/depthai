import argparse
import threading
import time

from depthai_demo import DepthAI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--streams", nargs="+")
    parser.add_argument("-cnn", "--cnn_model")
    args = parser.parse_args()

    dai = DepthAI()
    thread = threading.Thread(target=dai.startLoop, daemon=True)
    thread.start()

    start = time.time()
    timeout_s = 20
    missing_streams = set(filter(lambda stream: stream != "metaout", args.streams))
    data_success = False if len(missing_streams) > 0 else True
    nn_success = False if args.cnn_model is not None else True

    while time.time() - start < timeout_s:
        if dai.data_packets is not None and not data_success:
            missing_streams = missing_streams - set(map(lambda packet: packet.stream_name, dai.data_packets))
            if len(missing_streams) == 0:
                data_success = True
        if dai.nnet_packets is not None and not nn_success:
            if len(dai.nnet_packets) > 0:
                nn_success = True

        if data_success and nn_success:
            break

    success = data_success and nn_success
    dai.stopLoop()
    thread.join()
    del dai
    raise SystemExit(0 if success else 1)
