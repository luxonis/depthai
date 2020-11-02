#!/usr/bin/env python3

import json
import subprocess
import sys
from pip._internal.operations.freeze import freeze
import platform

try:
    import usb.core
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyusb"])
    import usb.core


def get_usb():
    speeds = ["Unknown", "Low", "Full", "High", "Super", "SuperPlus"]
    format_hex = lambda val: f"{val:#0{6}x}"
    try:
        for dev in usb.core.find(find_all=True):
            yield {
                "port": dev.port_number,
                "vendor_id": format_hex(dev.idVendor),
                "product_id": format_hex(dev.idProduct),
                "speed": speeds[dev.speed] if dev.speed < len(speeds) else dev.speed
            }
    except usb.core.NoBackendError:
        yield "No USB backend found"


data = {
    "architecture": ' '.join(platform.architecture()).strip(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python_build": ' '.join(platform.python_build()).strip(),
    "python_compiler": platform.python_compiler(),
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "release": platform.release(),
    "system": platform.system(),
    "version": platform.version(),
    "win32_ver": ' '.join(platform.win32_ver()).strip(),
    "uname": ' '.join(platform.uname()).strip(),
    "packages": list(freeze(local_only=True)),
    "usb": list(get_usb()),
}

with open("log_system_information.json", "w") as f:
    json.dump(data, f, indent=4)

print(json.dumps(data, indent=4))
print("System info gathered successfully - saved as \"log_system_information.json\"")
