import traceback
from pathlib import Path
from ..previews import Previews
import depthai as dai


class EncodingManager:
    def __init__(self, pm, encode_config: dict, encode_output=None):
        self.encoding_queues = {}
        self.encoding_nodes = {}
        self.encoding_files = {}
        self.encode_config = encode_config
        self.encode_output = Path(encode_output) or Path(__file__).parent
        self.pm = pm
        for camera_name, enc_fps in self.encode_config.items():
            self.create_encoder(camera_name, enc_fps)
            self.encoding_nodes[camera_name] = getattr(pm.nodes, camera_name + "_enc")

    def create_encoder(self, camera_name, enc_fps):
        allowed_sources = [Previews.left.name, Previews.right.name, Previews.color.name]
        if camera_name not in allowed_sources:
            raise ValueError(
                "Camera param invalid, received {}, available choices: {}".format(camera_name, allowed_sources))
        node_name = camera_name.lower() + '_enc'
        xout_name = node_name + "_xout"
        enc_profile = dai.VideoEncoderProperties.Profile.H264_MAIN

        if camera_name == Previews.color.name:
            if not hasattr(self.pm.nodes, 'cam_rgb'):
                raise RuntimeError("RGB camera not initialized. Call create_color_cam(res, fps) first!")
            enc_resolution = (self.pm.nodes.cam_rgb.getVideoWidth(), self.pm.nodes.cam_rgb.getVideoHeight())
            enc_profile = dai.VideoEncoderProperties.Profile.H265_MAIN
            enc_in = self.pm.nodes.cam_rgb.video

        elif camera_name == Previews.left.name:
            if not hasattr(self.pm.nodes, 'mono_left'):
                raise RuntimeError("Left mono camera not initialized. Call create_left_cam(res, fps) first!")
            enc_resolution = (
            self.pm.nodes.mono_left.getResolutionWidth(), self.pm.nodes.mono_left.getResolutionHeight())
            enc_in = self.pm.nodes.mono_left.out
        elif camera_name == Previews.right.name:
            if not hasattr(self.pm.nodes, 'mono_right'):
                raise RuntimeError("Right mono camera not initialized. Call create_right_cam(res, fps) first!")
            enc_resolution = (
            self.pm.nodes.mono_right.getResolutionWidth(), self.pm.nodes.mono_right.getResolutionHeight())
            enc_in = self.pm.nodes.mono_right.out
        else:
            raise NotImplementedError("Unable to create encoder for {]".format(camera_name))

        enc = self.pm.p.createVideoEncoder()
        enc.setDefaultProfilePreset(*enc_resolution, enc_fps, enc_profile)
        enc_in.link(enc.input)
        setattr(self.pm.nodes, node_name, enc)

        enc_xout = self.pm.p.createXLinkOut()
        enc.bitstream.link(enc_xout.input)
        enc_xout.setStreamName(xout_name)
        setattr(self.pm.nodes, xout_name, enc_xout)

    def create_default_queues(self, device):
        for camera_name, enc_fps in self.encode_config.items():
            self.encoding_queues[camera_name] = device.getOutputQueue(camera_name + "_enc_xout", maxSize=30,
                                                                      blocking=True)
            self.encoding_files[camera_name] = (self.encode_output / camera_name).with_suffix(
                ".h265" if self.encoding_nodes[
                               camera_name].getProfile() == dai.VideoEncoderProperties.Profile.H265_MAIN else ".h264"
            ).open('wb')

    def parse_queues(self):
        for name, queue in self.encoding_queues.items():
            while queue.has():
                queue.get().getData().tofile(self.encoding_files[name])

    def close(self):
        def print_manual():
            print(
                "To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
            cmd = "ffmpeg -framerate {} -i {} -c copy {}"
            for name, file in self.encoding_files.items():
                print(cmd.format(self.encoding_nodes[name].getFrameRate(), file.name,
                                 str(Path(file.name).with_suffix('.mp4'))))

        for name, file in self.encoding_files.items():
            file.close()
        try:
            import ffmpy3
            for name, file in self.encoding_files.items():
                fps = self.encoding_nodes[name].getFrameRate()
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
            for name, file in self.encoding_files.items():
                print("Produced file: {}".format(str(Path(file.name).with_suffix('.mp4'))))
        except ImportError:
            print("Module ffmpy3 not fouund!")
            traceback.print_exc()
            print_manual()
        except:
            print("Unknown error!")
            traceback.print_exc()
            print_manual()
