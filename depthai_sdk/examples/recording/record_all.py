from depthai_sdk import OakCamera, RecordType
from depthai_sdk.args_parser import ArgsParser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--recordStreams', action='store_true', help="Record frames to file")
parser.add_argument('--saveStreamsTo', type=str, help="Save frames to directory", default="./record")
args= ArgsParser.parseArgs(parser=parser)

with OakCamera(args=args) as oak:
    cams = oak.create_all_cameras()
    left = oak.camera('left')
    right = oak.camera('right')
    if left is not None and right is not None:
        stereo = oak.create_stereo(left=left, right=right)
        oak.visualize(stereo)
    # Sync & save all streams
    if args["recordStreams"]:
        oak.record(cams, args["saveStreamsTo"], RecordType.VIDEO_LOSSLESS).configure_syncing(True, 50)
    oak.visualize(cams, fps=True)

    oak.start(blocking=True)
