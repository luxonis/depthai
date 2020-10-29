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
    speeds = ["Unknown", "Low", "Full", "High", "Super"]
    format_hex = lambda val: f"{val:#0{6}x}"
    for dev in usb.core.find(find_all=True):
        yield {
            'port': dev.port_number,
            "vendor_id": format_hex(dev.idVendor),
            "product_id": format_hex(dev.idProduct),
            "speed": speeds[dev.speed]
        }


data = {
    "architecture": ' '.join(platform.architecture()).strip(),
    "libc": ' '.join(platform.libc_ver()).strip(),
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
    "win32_edition": platform.win32_edition(),
    "win32_ver": ' '.join(platform.win32_ver()).strip(),
    "uname": ' '.join(platform.uname()).strip(),
    "packages": list(freeze(local_only=True)),
    "usb": list(get_usb()),
}

with open("log_system_information.json", "w") as f:
    json.dump(data, f, indent=4)
