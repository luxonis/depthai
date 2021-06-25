#!/usr/bin/env python3
from pathlib import Path
import cv2
import depthai as dai

from depthai_helpers.managers import NNetManager, PreviewManager, FPSHandler, PipelineManager, Previews, EncodingManager
from depthai_helpers.version_check import check_depthai_version
import platform

from depthai_helpers.arg_manager import parse_args
from depthai_helpers.config_manager import BlobManager, ConfigManager
from depthai_helpers.utils import frame_norm, to_planar, to_tensor_result, load_module

print('Using depthai module from: ', dai.__file__)
print('Depthai version installed: ', dai.__version__)
if platform.machine() not in ['armv6l', 'aarch64']:
    check_depthai_version()

conf = ConfigManager(parse_args())
conf.linuxCheckApplyUsbRules()
if not conf.useCamera and str(conf.args.video).startswith('https'):
    conf.downloadYTVideo()
conf.adjustPreviewToOptions()

callbacks = load_module(conf.args.callback)
rgb_res = conf.getRgbResolution()
mono_res = conf.getMonoResolution()


disp_multiplier = 255 / conf.maxDisparity


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



device_info = conf.getDeviceInfo()
openvino_version = None
if conf.args.openvino_version:
    openvino_version = getattr(dai.OpenVINO.Version, 'VERSION_' + conf.args.openvino_version)
pm = PipelineManager(openvino_version)

input_size = tuple(map(int, conf.args.cnn_input_size.split('x'))) if conf.args.cnn_input_size else None

nn_manager = NNetManager(
    input_size=input_size,
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
    cap = cv2.VideoCapture(conf.args.video) if not conf.useCamera else None
    fps = FPSHandler() if conf.useCamera else FPSHandler(cap)

    if conf.useCamera or conf.args.sync:
        pv = PreviewManager(fps, display=conf.args.show, colorMap=conf.getColorMap(), disp_multiplier=disp_multiplier)

        if conf.args.camera == "left" or conf.useDepth:
            pm.create_left_cam(mono_res, conf.args.mono_fps)
        if conf.args.camera == "right" or conf.useDepth:
            pm.create_right_cam(mono_res, conf.args.mono_fps)
        if conf.args.camera == "color":
            pm.create_color_cam(rgb_res, conf.args.rgb_fps, conf.args.full_fov_nn, conf.useHQ)

        if conf.useDepth:
            pm.create_depth(
                conf.args.disparity_confidence_threshold,
                conf.getMedianFilter(),
                conf.args.stereo_lr_check,
                conf.args.extended_disparity,
                conf.args.subpixel,
            )

        enc_manager = EncodingManager(pm, dict(conf.args.encode), conf.args.encode_output) if len(conf.args.encode) > 0 else None

    if len(conf.args.report) > 0:
        pm.create_system_logger()

    nn_pipeline = nn_manager.create_nn_pipeline(pm.p, pm.nodes, shaves=conf.args.shaves, use_depth=conf.useDepth,
                                                use_sbb=conf.args.spatial_bounding_box and conf.useDepth,
                                                minDepth=conf.args.min_depth, maxDepth=conf.args.max_depth,
                                                sbbScaleFactor=conf.args.sbb_scale_factor)
    pm.create_nn(nn=nn_pipeline, sync=conf.args.sync)

    # Start pipeline
    device.startPipeline(pm.p)
    pm.create_default_queues(device)
    nn_in = device.getInputQueue(nn_manager.input_name, maxSize=1, blocking=False) if not conf.useCamera else None
    nn_out = device.getOutputQueue(nn_manager.output_name, maxSize=1, blocking=False)

    sbb_out = device.getOutputQueue("sbb", maxSize=1, blocking=False) if nn_manager.sbb else None
    log_out = device.getOutputQueue("system_logger", maxSize=30, blocking=False) if len(conf.args.report) > 0 else None

    if conf.useCamera:
        pv.create_queues(device)
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

                if sbb_out is not None and pv.has(Previews.depth.name):
                    sbb = sbb_out.tryGet()
                    if sbb is not None:
                        sbb_rois = sbb.getConfigData()
                    depth_frame = pv.get(Previews.depth.name)
                    for roi_data in sbb_rois:
                        roi = roi_data.roi.denormalize(depth_frame.shape[1], depth_frame.shape[0])
                        top_left = roi.topLeft()
                        bottom_right = roi.bottomRight()
                        # Display SBB on the disparity map
                        cv2.rectangle(pv.get("depth"), (int(top_left.x), int(top_left.y)), (int(bottom_right.x), int(bottom_right.y)), nn_manager.bbox_color[0], cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            else:
                read_correctly, host_frame = cap.read()
                if not read_correctly:
                    break

                scaled_frame = cv2.resize(host_frame, nn_manager.input_size)
                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(seq_num)
                frame_nn.setType(dai.RawImgFrame.Type.BGR888p)
                frame_nn.setWidth(nn_manager.input_size[0])
                frame_nn.setHeight(nn_manager.input_size[1])
                frame_nn.setData(to_planar(scaled_frame))
                nn_in.send(frame_nn)
                seq_num += 1

                # if high quality, send original frames
                if not conf.useHQ:
                    host_frame = scaled_frame
                fps.tick('host')

            in_nn = nn_out.tryGet()
            if in_nn is not None:
                callbacks.on_nn(in_nn)
                if not conf.useCamera and conf.args.sync:
                    host_frame = Previews.host.value(host_out.get())
                nn_data = nn_manager.decode(in_nn)
                fps.tick('nn')

            if conf.useCamera:
                nn_manager.draw(pv, nn_data)
                fps.draw_fps(pv)
                pv.show_frames(scale=conf.args.scale, callback=callbacks.on_show_frame)
            else:
                nn_manager.draw(host_frame, nn_data)
                fps.draw_fps(host_frame)
                cv2.imshow("host", host_frame)

            if log_out:
                logs = log_out.tryGetAll()
                for log in logs:
                    print_sys_info(log)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        if conf.useCamera and enc_manager is not None:
            enc_manager.close()

if conf.args.report_file:
    report_file.close()

callbacks.on_teardown(**locals())