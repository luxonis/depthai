import os
import json
import platform
import subprocess
from pathlib import Path
import consts.resource_paths

from depthai_helpers.model_downloader import download_model
from depthai_helpers.cli_utils import cli_print, PrintColors

class DepthConfigManager:
    labels = ""

    def __init__(self, args):
        self.args = args
        self.stream_list = args['streams']
        self.compile_model = self.args['shaves'] is not None and self.args['cmx_slices'] is not None and self.args['NN_engines']
        self.calc_dist_to_bb = not self.args['disable_depth']

        # Prepare handler methods (decode_nn, show_nn) for the network we want to run.
        self.decode_nn, self.show_nn, self.decode_nn_json = self.importAndGetCallbacksForNN()
        self.jsonConfig = self.generateJsonConfig()


    def getUsb2Mode(self):
        usb2_mode = False
        if self.args['force_usb2']:
            cli_print("FORCE USB2 MODE", PrintColors.WARNING)
            usb2_mode = True
        else:
            usb2_mode = False

    def getCommandFile(self):
        debug_mode = False
        cmd_file = ''
        if self.args['dev_debug'] == None:
            # Debug -debug flag NOT present,
            debug_mode = False
        elif self.args['dev_debug'] == '':
            # If just -debug flag is present -> cmd_file = '' (wait for device to be connected beforehand)
            debug_mode = True
        else:
            debug_mode = True
            cmd_file = self.args['dev_debug']

        return cmd_file, debug_mode


    def importAndGetCallbacksForNN(self):
        # why inline imports? Could just make a dict for all these to make managing them easier and less verbose.
        from depthai_helpers.mobilenet_ssd_handler import decode_mobilenet_ssd, show_mobilenet_ssd, decode_mobilenet_ssd_json
        decode_nn=decode_mobilenet_ssd
        show_nn=show_mobilenet_ssd
        decode_nn_json=decode_mobilenet_ssd_json

        if self.args['cnn_model'] == 'age-gender-recognition-retail-0013':
            from depthai_helpers.age_gender_recognition_handler import decode_age_gender_recognition, show_age_gender_recognition, decode_age_gender_recognition_json
            decode_nn=decode_age_gender_recognition
            show_nn=show_age_gender_recognition
            decode_nn_json=decode_age_gender_recognition_json
            self.calc_dist_to_bb=False

        if self.args['cnn_model'] == 'emotions-recognition-retail-0003':
            from depthai_helpers.emotion_recognition_handler import decode_emotion_recognition, show_emotion_recognition, decode_emotion_recognition_json
            decode_nn=decode_emotion_recognition
            show_nn=show_emotion_recognition
            decode_nn_json=decode_emotion_recognition_json
            self.calc_dist_to_bb=False

        if self.args['cnn_model'] == 'tiny-yolo':
            from depthai_helpers.tiny_yolo_v3_handler import decode_tiny_yolo, show_tiny_yolo, decode_tiny_yolo_json
            decode_nn=decode_tiny_yolo
            show_nn=show_tiny_yolo
            decode_nn_json=decode_tiny_yolo_json
            self.calc_dist_to_bb=False
            self.compile_model=False

        if self.args['cnn_model'] in ['facial-landmarks-35-adas-0002', 'landmarks-regression-retail-0009']:
            from depthai_helpers.landmarks_recognition_handler import decode_landmarks_recognition, show_landmarks_recognition, decode_landmarks_recognition_json
            decode_nn=decode_landmarks_recognition
            show_nn=show_landmarks_recognition
            decode_nn_json=decode_landmarks_recognition_json
            self.calc_dist_to_bb=False

        return decode_nn, show_nn, decode_nn_json


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


    def generateJsonConfig(self):

        # something to verify usb rules are good?
        self.linuxCheckApplyUsbRules()

        # left_right double NN check.
        if self.args['cnn_camera'] == 'left_right':
            if self.args['NN_engines'] is None:
                self.args['NN_engines'] = 2
                self.args['shaves'] = 6 if self.args['shaves'] is None else self.args['shaves'] - self.args['shaves'] % 2
                self.args['cmx_slices'] = 6 if self.args['cmx_slices'] is None else self.args['cmx_slices'] - self.args['cmx_slices'] % 2
                self.compile_model = True
                cli_print('Running NN on both cams requires 2 NN engines!', PrintColors.RED)

        # Get blob files
        blobMan = BlobManager(self.args, self.compile_model, self.calc_dist_to_bb)
        self.labels = blobMan.getLabels()
        if blobMan.default_blob:
            #default
            shave_nr = 7
            cmx_slices = 7
            NCE_nr = 1

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
                'padding_factor': 0.3,
                'depth_limit_m': 10.0, # In meters, for filtering purpose during x,y,z calc
                'confidence_threshold' : 0.5, #Depth is calculated for bounding boxes with confidence higher than this number
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

        config = self.postProcessJsonConfig(config)

        return config


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

        # check for known bad configurations
        if 'depth_sipp' in config['streams'] and ('depth_color_h' in config['streams'] or 'depth_mm_h' in config['streams']):
            print('ERROR: depth_sipp is mutually exclusive with depth_color_h')
            exit(2)
            # del config["streams"][config['streams'].index('depth_sipp')]

        # Append video stream if video recording was requested and stream is not already specified
        video_file = None
        if self.args['video'] is not None:
            
            # open video file
            try:
                video_file = open(self.args['video'], 'wb')
                if config['streams'].count('video') == 0:
                    config['streams'].append('video')
            except IOError:
                print("Error: couldn't open video file for writing. Disabled video output stream")
                if config['streams'].count('video') == 1:
                    config['streams'].remove('video')

        return config


class BlobManager:
    def __init__(self, args, compile_model, calc_dist_to_bb):
        self.args = args
        self.calc_dist_to_bb = calc_dist_to_bb

        if self.args['cnn_model']:
            self.blob_file, self.blob_file_config = self.getBlobFiles(self.args['cnn_model'])

        self.blob_file2 = ""
        self.blob_file_config2 = ""
        if self.args['cnn_model2']:
            print("Using CNN2:", self.args['cnn_model2'])
            self.blob_file2, self.blob_file_config2 = self.getBlobFiles(self.args['cnn_model2'], False)

        # verify the first blob files exist? I just copied this logic from before the refactor. Not sure if it's necessary. This makes it so this script won't run unless we have a blob file and config.
        self.verifyBlobFilesExist(self.blob_file, self.blob_file_config)

        # compile modules
        self.default_blob=True
        if compile_model:
            self.blob_file, self.default_blob = self.compileBlob(blob_file)
            if self.args['cnn_model2']:
                self.blob_file2, self.default_blob = self.compileBlob(blob_file2)

    def getLabels(self):
        # try and load labels
        with open(self.blob_file_config) as f:
            data = json.load(f)

        try:
            labels = data['mappings']['labels']
        except:
            labels = None
            print("Labels not found in json!")
    
        return labels

    def verifyBlobFilesExist(self, verifyBlob, verifyConfig):
        verifyBlobPath = Path(verifyBlob)
        verifyConfigPath = Path(verifyConfig)
        if not verifyBlobPath.exists():
            cli_print("\nWARNING: NN blob not found in: " + verifyBlob, PrintColors.WARNING)
            os._exit(1)

        if not verifyConfigPath.exists():
            cli_print("\nWARNING: NN json not found in: " + verifyConfig, PrintColors.WARNING)
            os._exit(1)

    def getBlobFiles(self, cnnModel, isFirstNN=True):
        cnn_model_path = consts.resource_paths.nn_resource_path + cnnModel + "/" + cnnModel
        blobFile = cnn_model_path + ".blob"
        suffix=""
        if self.calc_dist_to_bb and isFirstNN:
            suffix="_depth"
        blobFileConfig = cnn_model_path + suffix + ".json"

        self.verifyBlobFilesExist(blobFile, blobFileConfig)

        return blobFile, blobFileConfig

    def compileBlob(self, blob_file):
        default_blob=False
        shave_nr = self.args['shaves']
        cmx_slices = self.args['cmx_slices']
        NCE_nr = self.args['NN_engines']

        if NCE_nr == 2:
            if shave_nr % 2 == 1 or cmx_slices % 2 == 1:
                cli_print("shave_nr and cmx_slices config must be even number when NCE is 2!", PrintColors.RED)
                exit(2)
            shave_nr_opt = int(shave_nr / 2)
            cmx_slices_opt = int(cmx_slices / 2)
        else:
            shave_nr_opt = int(shave_nr)
            cmx_slices_opt = int(cmx_slices)

        outblob_file = blob_file + ".sh" + str(shave_nr) + "cmx" + str(cmx_slices) + "NCE" + str(NCE_nr)
        if(not Path(outblob_file).exists()):
            cli_print("Compiling model for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.RED)
            ret = download_model(self.args['cnn_model'], shave_nr_opt, cmx_slices_opt, NCE_nr, outblob_file)
            if(ret != 0):
                cli_print("Model compile failed. Falling back to default.", PrintColors.WARNING)
                default_blob=True
            else:
                blob_file = outblob_file
        else:
            cli_print("Compiled mode found: compiled for {0} shaves, {1} cmx_slices and {2} NN_engines ".format(str(shave_nr), str(cmx_slices), str(NCE_nr)), PrintColors.GREEN)
            blob_file = outblob_file

        return blob_file, default_blob
