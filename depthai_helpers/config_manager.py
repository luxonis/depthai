import os
import json
import platform
import subprocess
from pathlib import Path
import consts.resource_paths
import urllib.request

from depthai_helpers import utils
from model_compiler.model_compiler import download_and_compile_NN_model
from consts.resource_paths import nn_resource_path as model_zoo_folder
from depthai_helpers.cli_utils import cli_print, PrintColors

class DepthConfigManager:
    labels = ""
    NN_config = None
    custom_fw_commit = ''

    def __init__(self, args):
        self.args = args
        self.stream_list = args['streams']
        self.calc_dist_to_bb = not self.args['disable_depth']

        # Prepare handler methods (decode_nn, show_nn) for the network we want to run.
        self.importAndSetCallbacksForNN()
        self.generateJsonConfig()


    def getUsb2Mode(self):
        usb2_mode = False
        if self.args['force_usb2']:
            cli_print("FORCE USB2 MODE", PrintColors.WARNING)
            usb2_mode = True
        else:
            usb2_mode = False
        return usb2_mode

    def getColorPreviewScale(self):
        if self.args['color_scale']:
            return self.args['color_scale']
        else:
            return 1.0

    def getCustomFirmwarePath(self, commit):
        fwdir = '.fw_cache/'
        if not os.path.exists(fwdir):
            os.mkdir(fwdir)
        fw_variant = ''
        if self.getUsb2Mode():
            fw_variant = 'usb2-'
        fname = 'depthai-' + fw_variant + commit + '.cmd'
        path = fwdir + fname
        if not Path(path).exists():
            url = 'https://artifacts.luxonis.com/artifactory/luxonis-myriad-snapshot-local/depthai-device-side/'
            url += commit + '/' + fname
            print('Downloading custom FW:', url)
            # Need this to avoid "HTTP Error 403: Forbidden"
            class CustomURLopener(urllib.request.FancyURLopener):
                version = "Mozilla/5.0"
                # FancyURLopener doesn't report by default errors like 404
                def http_error_default(self, url, fp, errcode, errmsg, headers):
                    raise ValueError(errcode)
            url_opener = CustomURLopener()
            with url_opener.open(url) as response, open(path, 'wb') as outf:
                outf.write(response.read())
        return path

    def getCommandFile(self):
        debug_mode = False
        cmd_file = ''
        if self.args['firmware'] != None:
            self.custom_fw_commit = self.args['firmware']
        if self.args['dev_debug'] == None:
            # Debug -debug flag NOT present, check first for custom firmware
            if self.custom_fw_commit == '':
                debug_mode = False
            else:
                debug_mode = True
                cmd_file = self.getCustomFirmwarePath(self.custom_fw_commit)
        elif self.args['dev_debug'] == '':
            # If just -debug flag is present -> cmd_file = '' (wait for device to be connected beforehand)
            debug_mode = True
        else:
            debug_mode = True
            cmd_file = self.args['dev_debug']

        return cmd_file, debug_mode


    def importAndSetCallbacksForNN(self):
        # why inline imports? Could just make a dict for all these to make managing them easier and less verbose.
        from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd, decode_mobilenet_ssd_json
        self.decode_nn=decode_mobilenet_ssd
        self.show_nn=show_mobilenet_ssd
        self.decode_nn_json=decode_mobilenet_ssd_json

        if self.args['cnn_model'] == 'age-gender-recognition-retail-0013':
            from depthai_helpers.age_gender_recognition_handler import decode_age_gender_recognition, show_age_gender_recognition, decode_age_gender_recognition_json
            self.decode_nn=decode_age_gender_recognition
            self.show_nn=show_age_gender_recognition
            self.decode_nn_json=decode_age_gender_recognition_json
            self.calc_dist_to_bb=False

        if self.args['cnn_model'] == 'emotions-recognition-retail-0003':
            from depthai_helpers.emotion_recognition_handler import decode_emotion_recognition, show_emotion_recognition, decode_emotion_recognition_json
            self.decode_nn=decode_emotion_recognition
            self.show_nn=show_emotion_recognition
            self.decode_nn_json=decode_emotion_recognition_json
            self.calc_dist_to_bb=False

        if self.args['cnn_model'] in ['tiny-yolo-v3', 'yolo-v3']:
            from depthai_helpers.tiny_yolo_v3_handler import decode_tiny_yolo, show_tiny_yolo, decode_tiny_yolo_json
            self.decode_nn=decode_tiny_yolo
            self.show_nn=show_tiny_yolo
            self.decode_nn_json=decode_tiny_yolo_json

        if self.args['cnn_model'] in ['facial-landmarks-35-adas-0002', 'landmarks-regression-retail-0009']:
            from depthai_helpers.landmarks_recognition_handler import decode_landmarks_recognition, show_landmarks_recognition, decode_landmarks_recognition_json
            self.decode_nn=decode_landmarks_recognition
            self.show_nn=show_landmarks_recognition
            self.decode_nn_json=decode_landmarks_recognition_json
            self.calc_dist_to_bb=False

        # backward compatibility
        if self.args['cnn_model'] == 'openpose':
            self.args['cnn_model'] = 'human-pose-estimation-0001'
        if self.args['cnn_model'] == 'human-pose-estimation-0001':
            from depthai_helpers.openpose_handler import decode_openpose, show_openpose
            self.decode_nn=decode_openpose
            self.show_nn=show_openpose
            self.calc_dist_to_bb=False
            
        if self.args['cnn_model'] == 'openpose2':
            self.args['cnn_model'] = 'mobileNetV2-PoseEstimation'
        if self.args['cnn_model'] == 'mobileNetV2-PoseEstimation':
            from depthai_helpers.openpose2_handler import decode_openpose, show_openpose
            self.decode_nn=decode_openpose
            self.show_nn=show_openpose
            self.calc_dist_to_bb=False

        if self.args['cnn_model'] == 'deeplabv3p_person':
            from depthai_helpers.deeplabv3p_person import decode_deeplabv3p, show_deeplabv3p
            self.decode_nn=decode_deeplabv3p
            self.show_nn=show_deeplabv3p
            self.calc_dist_to_bb=False


    def linuxCheckApplyUsbRules(self):
        if platform.system() == 'Linux':
            ret = subprocess.call(['grep', '-irn', 'ATTRS{idVendor}=="03e7"', '/etc/udev/rules.d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if(ret != 0):
                cli_print("\nWARNING: Usb rules not found", PrintColors.WARNING)
                cli_print("\nSet rules: \n"
                """echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules \n"""
                "sudo udevadm control --reload-rules && udevadm trigger \n"
                "Disconnect/connect usb cable on host! \n", PrintColors.RED)
                os._exit(1)

    def getMaxShaveNumbers(self):
        stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in self.stream_list]
        max_shaves = 14
        if self.args['rgb_resolution'] != 1080:
            max_shaves = 11
            if 'object_tracker' in stream_names:
                max_shaves = 9
        elif 'object_tracker' in stream_names:
            max_shaves = 12

        return max_shaves

    def generateJsonConfig(self):

        # something to verify usb rules are good?
        self.linuxCheckApplyUsbRules()

        max_shave_nr = self.getMaxShaveNumbers()
        shave_nr = max_shave_nr if self.args['shaves'] is None else self.args['shaves']
        cmx_slices = shave_nr if self.args['cmx_slices'] is None else self.args['cmx_slices']
        NCE_nr = 1 if self.args['NN_engines'] is None else self.args['NN_engines']

        # left_right double NN check.
        if self.args['cnn_camera'] in ['left_right', 'rectified_left_right']:
            if NCE_nr != 2:
                cli_print('Running NN on both cams requires 2 NN engines!', PrintColors.RED)
                NCE_nr = 2

        if NCE_nr == 2:
            shave_nr = shave_nr - (shave_nr % 2)
            cmx_slices = cmx_slices - (cmx_slices % 2)

        # Get blob files
        blobMan = BlobManager(self.args, self.calc_dist_to_bb, shave_nr, cmx_slices, NCE_nr)
        self.NN_config = blobMan.getNNConfig()
        try:
            self.labels = self.NN_config['mappings']['labels']
        except:
            self.labels = None
            print("Labels not found in json!")

        try:
            output_format = self.NN_config['NN_config']['output_format']
        except:
            NN_json = {}
            NN_json['NN_config'] = {}
            NN_json['NN_config']['output_format'] = "raw"
            self.NN_config = NN_json
            output_format = "raw"

        if output_format == "raw" and self.calc_dist_to_bb == True:
            cli_print("WARNING: Depth calculation with raw output format is not supported! It's only supported for YOLO/mobilenet based NNs, disabling calc_dist_to_bb", PrintColors.WARNING)
            self.calc_dist_to_bb = False
        
        stream_names = [stream if isinstance(stream, str) else stream['name'] for stream in self.stream_list]
        if ('disparity' in stream_names or 'disparity_color' in stream_names) and self.calc_dist_to_bb == True:
            cli_print("WARNING: Depth calculation with disparity/disparity_color streams is not supported! Disabling calc_dist_to_bb", PrintColors.WARNING)
            self.calc_dist_to_bb = False

        # check for known bad configurations
        if 'depth' in stream_names and ('disparity' in stream_names or 'disparity_color' in stream_names):
            print('ERROR: depth is mutually exclusive with disparity/disparity_color')
            exit(2)

        if self.args['stereo_lr_check'] == True:
            raise ValueError("Left-right check option is still under development. Don;t enable it.")

        # Do not modify the default values in the config Dict below directly. Instead, use the `-co` argument when running this script.
        config = {
            # Possible streams:
            # ['left', 'right', 'jpegout', 'video', 'previewout', 'metaout', 'depth_sipp', 'disparity', 'depth_color_h']
            # If "left" is used, it must be in the first position.
            # To test depth use:
            # 'streams': [{'name': 'depth_sipp', "max_fps": 12.0}, {'name': 'previewout', "max_fps": 12.0}, ],
            'streams': self.args['streams'],
            'depth':
            {
                'calibration_file': consts.resource_paths.calib_fpath,
                'left_mesh_file': consts.resource_paths.left_mesh_fpath,
                'right_mesh_file': consts.resource_paths.right_mesh_fpath,
                'padding_factor': 0.3,
                'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
                'median_kernel_size': self.args['stereo_median_size'],
                'lr_check': self.args['stereo_lr_check'],
                'warp_rectify':
                {
                    'use_mesh' : self.args['use_mesh'], # if False, will use homography
                    'mirror_frame': self.args['mirror_rectified'] == 'true', # if False, the disparity will be mirrored instead
                    'edge_fill_color': 0, # gray 0..255, or -1 to replicate pixel values
                },
            },
            'ai':
            {
                'blob_file': blobMan.blob_file,
                'blob_file_config': blobMan.blob_file_config,
                'blob_file2': blobMan.blob_file2,
                'blob_file_config2': blobMan.blob_file_config2,
                'calc_dist_to_bb': self.calc_dist_to_bb,
                'keep_aspect_ratio': not self.args['full_fov_nn'],
                'camera_input': self.args['cnn_camera'],
                'shaves' : shave_nr,
                'cmx_slices' : cmx_slices,
                'NN_engines' : NCE_nr,
            },
            # object tracker
            'ot':
            {
                'max_tracklets'        : 20, #maximum 20 is supported
                'confidence_threshold' : 0.5, #object is tracked only for detections over this threshold
            },
            'board_config':
            {
                'swap_left_and_right_cameras': self.args['swap_lr'], # True for 1097 (RPi Compute) and 1098OBC (USB w/onboard cameras)
                'left_fov_deg': self.args['field_of_view'], # Same on 1097 and 1098OBC
                'rgb_fov_deg': self.args['rgb_field_of_view'],
                'left_to_right_distance_cm': self.args['baseline'], # Distance between stereo cameras
                'left_to_rgb_distance_cm': self.args['rgb_baseline'], # Currently unused
                'store_to_eeprom': self.args['store_eeprom'],
                'clear_eeprom': self.args['clear_eeprom'],
                'override_eeprom': self.args['override_eeprom'],
            },
            'camera':
            {
                'rgb':
                {
                    # 3840x2160, 1920x1080
                    # only UHD/1080p/30 fps supported for now
                    'resolution_h': self.args['rgb_resolution'],
                    'fps': self.args['rgb_fps'],
                },
                'mono':
                {
                    # 1280x720, 1280x800, 640x400 (binning enabled)
                    'resolution_h': self.args['mono_resolution'],
                    'fps': self.args['mono_fps'],
                },
            },
            'app':
            {
                'sync_video_meta_streams': self.args['sync_video_meta'],
                'sync_sequence_numbers'  : self.args['sync_sequence_numbers'],
                'usb_chunk_KiB' : self.args['usb_chunk_KiB'],
            },
            #'video_config':
            #{
            #    'rateCtrlMode': 'cbr', # Options: cbr / vbr
            #    'profile': 'h265_main', # Options: 'h264_baseline' / 'h264_main' / 'h264_high' / 'h265_main / 'mjpeg' '
            #    'bitrate': 8000000, # When using CBR (H264/H265 only)
            #    'maxBitrate': 8000000, # When using CBR (H264/H265 only)
            #    'keyframeFrequency': 30, (H264/H265 only)
            #    'numBFrames': 0, (H264/H265 only)
            #    'quality': 80 # (0 - 100%) When using VBR or MJPEG profile
            #}
            #'video_config':
            #{
            #    'profile': 'mjpeg',
            #    'quality': 95
            #}
        }

        self.jsonConfig = self.postProcessJsonConfig(config)


    def postProcessJsonConfig(self, config):
        # merge board config, if exists
        if self.args['board']:
            board_path = Path(self.args['board'])
            if not board_path.exists():
                board_path = Path(consts.resource_paths.boards_dir_path) / Path(self.args['board'].upper()).with_suffix('.json')
                if not board_path.exists():
                    print('ERROR: Board config not found: {}'.format(board_path))
                    os._exit(2)
            with open(board_path) as fp:
                board_config = json.load(fp)
            utils.merge(board_config, config)

        # handle config overwrite option.
        if self.args['config_overwrite']:
            self.args['config_overwrite'] = json.loads(self.args['config_overwrite'])
            config = utils.merge(self.args['config_overwrite'],config)
            print("Merged Pipeline config with overwrite",config)

        # Append video stream if video recording was requested and stream is not already specified
        self.video_file = None
        if self.args['video'] is not None:
            
            # open video file
            try:
                self.video_file = open(self.args['video'], 'wb')
                if config['streams'].count('video') == 0:
                    config['streams'].append('video')
            except IOError:
                print("Error: couldn't open video file for writing. Disabled video output stream")
                if config['streams'].count('video') == 1:
                    config['streams'].remove('video')

        return config



class BlobManager:
    def __init__(self, args, calc_dist_to_bb, shave_nr, cmx_slices, NCE_nr):
        self.args = args
        self.calc_dist_to_bb = calc_dist_to_bb
        self.shave_nr = shave_nr
        self.cmx_slices = cmx_slices
        self.NCE_nr = NCE_nr

        if self.args['cnn_model']:
            self.blob_file, self.blob_file_config = self.getBlobFiles(self.args['cnn_model'])

        self.blob_file2 = ""
        self.blob_file_config2 = ""
        if self.args['cnn_model2']:
            print("Using CNN2:", self.args['cnn_model2'])
            self.blob_file2, self.blob_file_config2 = self.getBlobFiles(self.args['cnn_model2'], False)

        # compile models
        self.blob_file = self.compileBlob(self.args['cnn_model'], self.args['model_compilation_target'])
        if self.args['cnn_model2']:
            self.blob_file2 = self.compileBlob(self.args['cnn_model2'], self.args['model_compilation_target'])

        # verify the first blob files exist? I just copied this logic from before the refactor. Not sure if it's necessary. This makes it so this script won't run unless we have a blob file and config.
        self.verifyBlobFilesExist(self.blob_file, self.blob_file_config)

    def getNNConfig(self):
        # try and load labels
        NN_json = None
        if Path(self.blob_file_config).exists():
            with open(self.blob_file_config) as f:
                if f is not None:
                    NN_json = json.load(f)
                    f.close()
        
        return NN_json

    def verifyBlobFilesExist(self, verifyBlob, verifyConfig):
        verifyBlobPath = Path(verifyBlob)
        verifyConfigPath = Path(verifyConfig)
        if not verifyBlobPath.exists():
            cli_print("\nWARNING: NN blob not found in: " + verifyBlob, PrintColors.WARNING)
            os._exit(1)

        if not verifyConfigPath.exists():
            print("NN config not found in: " + verifyConfig + '. Defaulting to "raw" output format!')

    def getBlobFiles(self, cnnModel, isFirstNN=True):
        cnn_model_path = consts.resource_paths.nn_resource_path + cnnModel + "/" + cnnModel
        blobFile = cnn_model_path + ".blob"
        blobFileConfig = cnn_model_path + ".json"

        return blobFile, blobFileConfig

    def compileBlob(self, nn_model, model_compilation_target):
        blob_file, _ = self.getBlobFiles(nn_model)

        shave_nr = self.shave_nr
        cmx_slices = self.cmx_slices
        NCE_nr = self.NCE_nr

        if NCE_nr == 2:
            if shave_nr % 2 == 1 or cmx_slices % 2 == 1:
                raise ValueError("shave_nr and cmx_slices config must be even number when NCE is 2!")
            shave_nr_opt = int(shave_nr / 2)
            cmx_slices_opt = int(cmx_slices / 2)
        else:
            shave_nr_opt = int(shave_nr)
            cmx_slices_opt = int(cmx_slices)

        outblob_file = blob_file + ".sh" + str(shave_nr) + "cmx" + str(cmx_slices) + "NCE" + str(NCE_nr)
        if(not Path(outblob_file).exists()):
            cli_print("Compiling model for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.RED)
            ret = download_and_compile_NN_model(nn_model, model_zoo_folder, shave_nr_opt, cmx_slices_opt, NCE_nr, outblob_file, model_compilation_target)
            if(ret != 0):
                cli_print("Model compile failed. Falling back to default.", PrintColors.WARNING)
                raise RuntimeError("Model compilation failed! Not connected to the internet?")
            else:
                blob_file = outblob_file
        else:
            cli_print("Compiled mode found: compiled for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.GREEN)
            blob_file = outblob_file

        return blob_file
