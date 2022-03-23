#!/usr/bin/env python3

import json
import platform


def make_sys_report(anonymous=False, skipUsb=False, skipPackages=False):
    def get_usb():
        try:
            import usb.core
        except ImportError:
            yield "NoLib"
            return
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

    result = {
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
    }

    if not skipPackages:
        from pip._internal.operations.freeze import freeze
        result["packages"] = list(freeze(local_only=True))
    if not skipUsb:
        result["usb"] = list(get_usb())
    if not anonymous:
        result["uname"] = ' '.join(platform.uname()).strip(),
    return result


if __name__ == "__main__":
    data = make_sys_report()
    with open("log_system_information.json", "w") as f:
        json.dump(data, f, indent=4)

    print(json.dumps(data, indent=4))
    print("System info gathered successfully - saved as \"log_system_information.json\"")
