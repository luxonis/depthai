#!/usr/bin/env python3
import os
from itertools import cycle
from pathlib import Path
import cv2
import depthai as dai

from depthai_helpers.managers import NNetManager, PreviewManager, FPSHandler, PipelineManager, Previews, EncodingManager
from depthai_helpers.version_check import check_depthai_version
import platform

from depthai_helpers.arg_manager import parse_args
from depthai_helpers.config_manager import ConfigManager
from depthai_helpers.utils import frame_norm, to_planar, to_tensor_result, load_module

DISP_CONF_MIN = int(os.getenv("DISP_CONF_MIN", 0))
DISP_CONF_MAX = int(os.getenv("DISP_CONF_MAX", 255))
SIGMA_MIN = int(os.getenv("SIGMA_MIN", 0))
SIGMA_MAX = int(os.getenv("SIGMA_MAX", 65535))
LRCT_MIN = int(os.getenv("LRCT_MIN", 0))
LRCT_MAX = int(os.getenv("LRCT_MAX", 10))

print('Using depthai module from: ', dai.__file__)
print('Depthai version installed: ', dai.__version__)
if platform.machine() not in ['armv6l', 'aarch64']:
    check_depthai_version()

conf = ConfigManager(parse_args())
conf.linuxCheckApplyUsbRules()
if not conf.useCamera and str(conf.args.video).startswith('https'):
    conf.downloadYTVideo()

callbacks = load_module(conf.args.callback)
rgb_res = conf.getRgbResolution()
mono_res = conf.getMonoResolution()


dispMultiplier = 255 / conf.maxDisparity


if conf.args.report_file:
    report_file_p = Path(conf.args.report_file).with_suffix('.csv')
    report_file_p.parent.mkdir(parents=True, exist_ok=True)
    report_file = open(conf.args.report_file, 'a')

def print_sys_info(info):
    m = 1024 * 1024 # MiB
    if not conf.args.report_file:
        if "memory" in conf.args.report:
            print(f"Drr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB")
            print(f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB")
            print(f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB")
            print(f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB")
        if "temp" in conf.args.report:
            t = info.chipTemperature
            print(f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa0: {t.upa:.2f}, upa1: {t.dss:.2f}")
        if "cpu" in conf.args.report:
            print(f"Cpu usage - Leon OS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon RT: {info.leonMssCpuUsage.average * 100:.2f} %")
        print("----------------------------------------")
    else:
        data = {}
        if "memory" in conf.args.report:
            data = {
                **data,
                "ddr_used": info.ddrMemoryUsage.used,
                "ddr_total": info.ddrMemoryUsage.total,
                "cmx_used": info.cmxMemoryUsage.used,
                "cmx_total": info.cmxMemoryUsage.total,
                "leon_css_used": info.leonCssMemoryUsage.used,
                "leon_css_total": info.leonCssMemoryUsage.total,
                "leon_mss_used": info.leonMssMemoryUsage.used,
                "leon_mss_total": info.leonMssMemoryUsage.total,
            }
        if "temp" in conf.args.report:
            data = {
                **data,
                "temp_avg": info.chipTemperature.average,
                "temp_css": info.chipTemperature.css,
                "temp_mss": info.chipTemperature.mss,
                "temp_upa0": info.chipTemperature.upa,
                "temp_upa1": info.chipTemperature.dss,
            }
        if "cpu" in conf.args.report:
            data = {
                **data,
                "cpu_css_avg": info.leonCssCpuUsage.average,
                "cpu_mss_avg": info.leonMssCpuUsage.average,
            }

        if report_file.tell() == 0:
            print(','.join(data.keys()), file=report_file)
        callbacks.on_report(data)
        print(','.join(map(str, data.values())), file=report_file)


class Trackbars:
    instances = {}

    @staticmethod
    def create_trackbar(name, window, min_val, max_val, default_val, callback):
        def fn(value):
            if Trackbars.instances[name][window] != value:
                callback(value)
            for other_window, previous_value in Trackbars.instances[name].items():
                if other_window != window and previous_value != value:
                    Trackbars.instances[name][other_window] = value
                    cv2.setTrackbarPos(name, other_window, value)

        cv2.createTrackbar(name, window, min_val, max_val, fn)
        Trackbars.instances[name] = {**Trackbars.instances.get(name, {}), window: default_val}
        cv2.setTrackbarPos(name, window, default_val)

device_info = conf.getDeviceInfo()
openvino_version = None
if conf.args.openvino_version:
    openvino_version = getattr(dai.OpenVINO.Version, 'VERSION_' + conf.args.openvino_version)
pm = PipelineManager(openvino_version)

if conf.useNN:
    nn_manager = NNetManager(
        input_size=conf.inputSize,
        model_name=conf.getModelName(),
        model_dir=conf.getModelDir(),
        source=conf.getModelSource(),
        full_fov=conf.args.full_fov_nn or not conf.useCamera,
        flip_detection=conf.getModelSource() in ("rectified_left", "rectified_right") and not conf.args.stereo_lr_check
    )
    nn_manager.count_label = conf.getCountLabel(nn_manager)
    pm.set_nn_manager(nn_manager)

# Pipeline is defined, now we can connect to the device
with dai.Device(pm.p.getOpenVINOVersion(), device_info, usb2Mode=conf.args.usb_speed == "usb2") as device:
    conf.adjustParamsToDevice(device)
    conf.adjustPreviewToOptions()
    if conf.lowBandwidth:
        pm.enableLowBandwidth()
    cap = cv2.VideoCapture(conf.args.video) if not conf.useCamera else None
    fps = FPSHandler() if conf.useCamera else FPSHandler(cap)

    if conf.useCamera or conf.args.sync:
        pv = PreviewManager(fps, display=conf.args.show, colorMap=conf.getColorMap(), dispMultiplier=dispMultiplier, mouseTracker=True, lowBandwidth=conf.lowBandwidth, scale=conf.args.scale)

        if conf.leftCameraEnabled:
            pm.create_left_cam(mono_res, conf.args.mono_fps, xout=Previews.left.name in conf.args.show)
        if conf.rightCameraEnabled:
            pm.create_right_cam(mono_res, conf.args.mono_fps, xout=Previews.right.name in conf.args.show)
        if conf.rgbCameraEnabled:
            pm.create_color_cam(nn_manager.input_size if conf.useNN else conf.previewSize, rgb_res, conf.args.rgb_fps, conf.args.full_fov_nn, xout=Previews.color.name in conf.args.show)

        if conf.useDepth:
            pm.create_depth(
                conf.args.disparity_confidence_threshold,
                conf.getMedianFilter(),
                conf.args.sigma,
                conf.args.stereo_lr_check,
                conf.args.lrc_threshold,
                conf.args.extended_disparity,
                conf.args.subpixel,
                useDepth=Previews.depth.name in conf.args.show or Previews.depth_raw.name in conf.args.show,
                useDisparity=Previews.disparity.name in conf.args.show or Previews.disparity_color.name in conf.args.show,
                useRectifiedLeft=Previews.rectified_left.name in conf.args.show,
                useRectifiedRight=Previews.rectified_right.name in conf.args.show,
            )

        enc_manager = EncodingManager(pm, conf.args.encode, conf.args.encode_output) if len(conf.args.encode) > 0 else None

    if len(conf.args.report) > 0:
        pm.create_system_logger()

    if conf.useNN:
        nn_pipeline = nn_manager.create_nn_pipeline(pm.p, pm.nodes, shaves=conf.args.shaves, use_depth=conf.useDepth,
                                                    use_sbb=conf.args.spatial_bounding_box and conf.useDepth,
                                                    minDepth=conf.args.min_depth, maxDepth=conf.args.max_depth,
                                                    sbbScaleFactor=conf.args.sbb_scale_factor)

        pm.create_nn(nn=nn_pipeline, sync=conf.args.sync, xout_nn_input=Previews.nn_input.name in conf.args.show,
                     xout_sbb=conf.args.spatial_bounding_box and conf.useDepth)

    # Start pipeline
    device.startPipeline(pm.p)
    pm.create_default_queues(device)
    nn_in = device.getInputQueue(nn_manager.input_name, maxSize=1, blocking=False) if not conf.useCamera and conf.useNN else None
    nn_out = device.getOutputQueue(nn_manager.output_name, maxSize=1, blocking=False) if conf.useNN else None

    sbb_out = device.getOutputQueue("sbb", maxSize=1, blocking=False) if conf.useNN and nn_manager.sbb else None
    log_out = device.getOutputQueue("system_logger", maxSize=30, blocking=False) if len(conf.args.report) > 0 else None

    median_filters = cycle([item for name, item in vars(dai.MedianFilter).items() if name.startswith('KERNEL_') or name.startswith('MEDIAN_')])
    for med_filter in median_filters:
        # move the cycle to the current median filter
        if med_filter == pm.depthConfig.getMedianFilter():
            break

    if conf.useCamera:
        def create_queue_callback(queue_name):
            if queue_name in [Previews.disparity_color.name, Previews.disparity.name, Previews.depth.name, Previews.depth_raw.name]:
                Trackbars.create_trackbar('Disparity confidence', queue_name, DISP_CONF_MIN, DISP_CONF_MAX, conf.args.disparity_confidence_threshold,
                         lambda value: pm.update_depth_config(device, dct=value))
                if queue_name in [Previews.depth_raw.name, Previews.depth.name]:
                    Trackbars.create_trackbar('Bilateral sigma', queue_name, SIGMA_MIN, SIGMA_MAX, conf.args.sigma,
                             lambda value: pm.update_depth_config(device, sigma=value))
                if conf.args.stereo_lr_check:
                    Trackbars.create_trackbar('LR-check threshold', queue_name, LRCT_MIN, LRCT_MAX, conf.args.lrc_threshold,
                             lambda value: pm.update_depth_config(device, lrc_threshold=value))
        pv.create_queues(device, create_queue_callback)
        if enc_manager is not None:
            enc_manager.create_default_queues(device)
    elif conf.args.sync:
        host_out = device.getOutputQueue(Previews.host.name, maxSize=1, blocking=False)

    seq_num = 0
    host_frame = None
    nn_data = []
    sbb_rois = []
    callbacks.on_setup(**locals())

    try:
        while True:
            fps.next_iter()
            callbacks.on_iter(**locals())
            if conf.useCamera:
                pv.prepare_frames(callback=callbacks.on_new_frame)
                if enc_manager is not None:
                    enc_manager.parse_queues()

                if sbb_out is not None:
                    sbb = sbb_out.tryGet()
                    if sbb is not None:
                        sbb_rois = sbb.getConfigData()
                    depth_frames = [pv.get(Previews.depth_raw.name), pv.get(Previews.depth.name)]
                    for depth_frame in depth_frames:
                        if depth_frame is None:
                            continue

                        for roi_data in sbb_rois:
                            roi = roi_data.roi.denormalize(depth_frame.shape[1], depth_frame.shape[0])
                            top_left = roi.topLeft()
                            bottom_right = roi.bottomRight()
                            # Display SBB on the disparity map
                            cv2.rectangle(depth_frame, (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), nn_manager.bbox_color[0], cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            else:
                read_correctly, host_frame = cap.read()
                if not read_correctly:
                    break

                if nn_in is not None:
                    scaled_frame = cv2.resize(host_frame, nn_manager.input_size)
                    frame_nn = dai.ImgFrame()
                    frame_nn.setSequenceNum(seq_num)
                    frame_nn.setType(dai.ImgFrame.Type.BGR888p)
                    frame_nn.setWidth(nn_manager.input_size[0])
                    frame_nn.setHeight(nn_manager.input_size[1])
                    frame_nn.setData(to_planar(scaled_frame))
                    nn_in.send(frame_nn)
                    seq_num += 1
                fps.tick('host')

            if nn_out is not None:
                in_nn = nn_out.tryGet()
                if in_nn is not None:
                    callbacks.on_nn(in_nn)
                    if not conf.useCamera and conf.args.sync:
                        host_frame = Previews.host.value(host_out.get())
                    nn_data = nn_manager.decode(in_nn)
                    fps.tick('nn')

            if conf.useCamera:
                if conf.useNN:
                    nn_manager.draw(pv, nn_data)

                def show_frames_callback(frame, name):
                    fps.draw_fps(frame, name)
                    if name in [Previews.disparity_color.name, Previews.disparity.name, Previews.depth.name, Previews.depth_raw.name]:
                        h, w = frame.shape[:2]
                        text = "Median filter: {} [M]".format(pm.depthConfig.getMedianFilter().name.lstrip("KERNEL_").lstrip("MEDIAN_"))
                        text_config = {
                            "fontFace": cv2.FONT_HERSHEY_TRIPLEX,
                            "fontScale": 0.4,
                            "thickness": 1
                        }
                        cv2.putText(frame, text, (10, h - 10), fps.fps_type, 0.5, fps.fps_bg_color, 4, fps.fps_line_type)
                        cv2.putText(frame, text, (10, h - 10), fps.fps_type, 0.5, fps.fps_color, 1, fps.fps_line_type)
                        return_frame = callbacks.on_show_frame(frame, name)
                        return return_frame if return_frame is not None else frame
                pv.show_frames(callback=show_frames_callback)
            else:
                if conf.useNN:
                    nn_manager.draw(host_frame, nn_data)
                fps.draw_fps(host_frame, "host")
                cv2.imshow("host", host_frame)

            if log_out:
                logs = log_out.tryGetAll()
                for log in logs:
                    print_sys_info(log)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('m'):
                next_filter = next(median_filters)
                pm.update_depth_config(device, median=next_filter)
    finally:
        if conf.useCamera and enc_manager is not None:
            enc_manager.close()

if conf.args.report_file:
    report_file.close()

fps.print_status()
callbacks.on_teardown(**locals())