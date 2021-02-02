#!/usr/bin/env python3

import subprocess
import sys
import platform
from pathlib import Path

platforms_requires_build_from_source = ["armv6l"]

# https://stackoverflow.com/a/58026969/5494277
in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_install = pip_call + ["install"]
if platform.machine() in platforms_requires_build_from_source:
    pip_install = pip_install + ["--force-reinstall"]

if not in_venv:
    pip_install.append("--user")

subprocess.check_call([*pip_install, "pip", "-U"])
# temporary workaroud for issue between main and develop
subprocess.check_call([*pip_call, "uninstall", "depthai", "--yes"])
subprocess.check_call([*pip_install, "-r", "requirements.txt"])
if platform.machine() in platforms_requires_build_from_source:
    import requests
    import os

    def download_file_from_google_drive(id, destination):
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        save_response_content(response, destination)    

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    print(f"Installing wheel for {platform.machine()} platform")
    directory = Path(".aux_wheels/")
    directory.mkdir(parents=True, exist_ok=True)
    wheel_dest = directory / Path("depthai-0.4.1.1-cp37-cp37m-linux_armv6l.whl")
    download_file_from_google_drive('16nq1sY2-MkgNZcRhUnndTTH5WlUNM5jS', wheel_dest)
    subprocess.check_call([*pip_install, str(wheel_dest)])

try:
    subprocess.check_call([*pip_install, "-r", "requirements-optional.txt"])
except subprocess.CalledProcessError as ex:
    print(f"Optional dependencies were not installed (exit code {ex.returncode})")
