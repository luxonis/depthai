from datetime import datetime

import pyrebase
import depthai as dai

from log_system_information import make_sys_report


class MetricManager:
    def __init__(self):
        self.config = {
          "apiKey": "AIzaSyDCrQs4SYUXZiN1qASxaiMU33YKSImp6kw",
          "authDomain": "depthai-data.firebaseapp.com",
          "databaseURL": "https://depthai-data-default-rtdb.firebaseio.com/",
          "storageBucket": "depthai-data.appspot.com"
        }

        self.firebase = pyrebase.initialize_app(self.config)
        self.db = self.firebase.database()
        self.demo_table = self.db.child("demo")

    def reportDevice(self, device: dai.Device):
        try:
            device_info = device.getDeviceInfo()
            mxid = device.getMxId()
        except:
            return
        try:
            cameras = list(map(lambda camera: camera.name, device.getConnectedCameras()))
        except:
            cameras = []
        try:
            usb = device.getUsbSpeed().name
        except:
            usb = "None"
        sys_report = make_sys_report(anonymous=True, skipUsb=True)
        data = {
            "mxid": mxid,
            "timestamp": datetime.utcnow().isoformat(),
            "system": sys_report,
            "api": {
                "version": dai.__version__
            },
            "device": {
                "cameras": cameras,
                "state": device_info.state.name,
                "platform": device_info.desc.platform.name,
                "protocol": device_info.desc.protocol.name,
                "usb": usb
            }
        }
        try:
            self.demo_table.update({
                f"devices/{mxid}": data
            })
        except:
            pass

if __name__ == "__main__":
    mm = MetricManager()
    with dai.Device() as device:
        mm.reportDevice(device)