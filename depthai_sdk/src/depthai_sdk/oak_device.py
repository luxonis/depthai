from typing import List, Dict, Any

import depthai as dai

from depthai_sdk.oak_outputs.xout_base import XoutBase


class OakDevice:
    device: dai.Device = None
    # fpsHandlers: Dict[str, FPS] = dict()
    oak_out_streams: List[XoutBase] = []

    @property
    def image_sensors(self) -> List[dai.CameraBoardSocket]:
        """
        Available imageSensors available on the camera
        """
        return self.device.getConnectedCameras()

    @property
    def info(self) -> dai.DeviceInfo:
        return self.device.getDeviceInfo()

    def stats_report(self) -> Dict[str, Any]:
        stats = {'mxid': self.device.getMxId()}

        css_cpu_usage = self.device.getLeonCssCpuUsage().average
        mss_cpu_usage = self.device.getLeonMssCpuUsage().average
        cmx_mem_usage = self.device.getCmxMemoryUsage()
        ddr_mem_usage = self.device.getDdrMemoryUsage()
        chip_temp = self.device.getChipTemperature()

        stats['css_usage'] = int(100 * css_cpu_usage)
        stats['mss_usage'] = int(100 * mss_cpu_usage)
        stats['ddr_mem_free'] = int(ddr_mem_usage.total - ddr_mem_usage.used)
        stats['ddr_mem_total'] = int(ddr_mem_usage.total)
        stats['cmx_mem_free'] = int(cmx_mem_usage.total - cmx_mem_usage.used)
        stats['cmx_mem_total'] = int(cmx_mem_usage.total)
        stats['css_temp'] = int(100 * chip_temp.css)
        stats['mss_temp'] = int(100 * chip_temp.mss)
        stats['upa_temp'] = int(100 * chip_temp.upa)
        stats['dss_temp'] = int(100 * chip_temp.dss)
        stats['temp'] = int(100 * chip_temp.average)

        return stats

    def info_report(self) -> Dict[str, Any]:
        """Returns device info"""
        info = {
            'mxid': self.device.getMxId(),
            'protocol': 'unknown',
            'platform': 'unknown',
            'product_name': 'unknown',
            'board_name': 'unknown',
            'board_rev': 'unknown',
            'bootloader_version': 'unknown',
        }
        try:
            device_info = self.device.getDeviceInfo()
        except:
            device_info = None

        try:
            eeprom_data = self.device.readFactoryCalibration().getEepromData()
        except:
            try:
                eeprom_data = self.device.readCalibration2().getEepromData()
            except:
                eeprom_data = None  # Could be due to some malfunction with the device, or simply device is disconnected currently.

        if eeprom_data:
            info['product_name'] = eeprom_data.productName
            info['board_name'] = eeprom_data.boardName
            info['board_rev'] = eeprom_data.boardRev
            info['bootloader_version'] = str(eeprom_data.version)

        if device_info:
            info['protocol'] = device_info.protocol.name
            info['platform'] = device_info.platform.name

        return info


    def init_callbacks(self, pipeline: dai.Pipeline):
        for node in pipeline.getAllNodes():
            if isinstance(node, dai.node.XLinkOut):
                stream_name = node.getStreamName()
                # self.fpsHandlers[name] = FPS()
                self.device.getOutputQueue(stream_name, maxSize=4, blocking=False).addCallback(
                    lambda name, msg: self.new_msg(name, msg)
                )

    def new_msg(self, name, msg):
        for sync in self.oak_out_streams:
            sync.new_msg(name, msg)

    def check_sync(self):
        """
        Checks whether there are new synced messages, non-blocking.
        """
        for sync in self.oak_out_streams:
            sync.check_queue(block=False)  # Don't block!
