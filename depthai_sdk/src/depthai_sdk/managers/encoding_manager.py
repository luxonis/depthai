import traceback
from pathlib import Path
from ..previews import Previews
import depthai as dai


class EncodingManager:
    """
    Manager class that handles video encoding
    """

    _encoding_queues = {}
    _encoding_nodes = {}
    _encoding_files = {}

    def __init__(self, encode_config: dict, encode_output=None):
        """
        Args:
            encode_config (dict): Encoding config consisting of keys as preview names and values being the encoding FPS
            encode_output (pathlib.Path): Output directory for the recorded videos
        """
        self.encode_config = encode_config
        self.encode_output = Path(encode_output) or Path(__file__).parent

    def create_encoders(self, pm):
        """
        Creates VideoEncoder nodes using Pipeline manager, based on config provided during initialization

        Args:
            pm (depthai_sdk.managers.PipelineManager): Pipeline Manager instance
        """

        self._encoding_nodes.clear()
        for camera_name, enc_fps in self.encode_config.items():
            pm.create_encoder(camera_name, enc_fps)
            self._encoding_nodes[camera_name] = getattr(pm.nodes, camera_name + "_enc")

    def create_default_queues(self, device):
        """
        Creates output queues for VideoEncoder nodes created in :code:`create_encoders` function. Also, opems up
        the H.264 / H.265 stream files (e.g. :code:`color.h265`) where the encoded data will be stored.

        Args:
            device (depthai.Device): Running device instance
        """
        self._encoding_queues.clear()
        self._encoding_files.clear()

        for camera_name, node in self._encoding_nodes.items():
            self._encoding_queues[camera_name] = device.getOutputQueue(camera_name + "_enc_xout", maxSize=30, blocking=True)
            suffix = ".h265" if node.getProfile() == dai.VideoEncoderProperties.Profile.H265_MAIN else ".h264"
            self._encoding_files[camera_name] = (self.encode_output / camera_name).with_suffix(suffix).open('wb')

    def parse_queues(self):
        """
        Parse the output queues, consuming the available data packets in it and storing them inside opened stream files
        """
        for name, queue in self._encoding_queues.items():
            while queue.has():
                queue.get().getData().tofile(self._encoding_files[name])

    def close(self):
        """
        Closes opened stream files and tries to perform FFMPEG-based conversion from raw stream into mp4 video.

        If successful, each stream file (e.g. :code:`color.h265`) will be available along with a ready to use video file
        (e.g. :code:`color.mp4`).

        In case of failure, this method will print traceback and commands that can be used for manual conversion
        """

        def print_manual():
            print(
                "To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
            cmd = "ffmpeg -framerate {} -i {} -c copy {}"
            for name, file in self._encoding_files.items():
                print(cmd.format(self._encoding_nodes[name].getFrameRate(), file.name, str(Path(file.name).with_suffix('.mp4'))))

        for name, file in self._encoding_files.items():
            file.close()
        try:
            import ffmpy3
            for name, file in self._encoding_files.items():
                fps = self._encoding_nodes[name].getFrameRate()
                out_name = str(Path(file.name).with_suffix('.mp4'))
                try:
                    ff = ffmpy3.FFmpeg(
                        inputs={file.name: "-y"},
                        outputs={out_name: "-c copy -framerate {}".format(fps)}
                    )
                    print("Running conversion command... [{}]".format(ff.cmd))
                    ff.run()
                except ffmpy3.FFExecutableNotFoundError:
                    print("FFMPEG executable not found!")
                    traceback.print_exc()
                    print_manual()
                except ffmpy3.FFRuntimeError:
                    print("FFMPEG runtime error!")
                    traceback.print_exc()
                    print_manual()
            print("Video conversion complete!")
            for name, file in self._encoding_files.items():
                print("Produced file: {}".format(str(Path(file.name).with_suffix('.mp4'))))
        except ImportError:
            print("Module ffmpy3 not fouund!")
            traceback.print_exc()
            print_manual()
        except:
            print("Unknown error!")
            traceback.print_exc()
            print_manual()
