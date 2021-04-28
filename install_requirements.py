#!/usr/bin/env python3

import subprocess
import sys
import platform
from pathlib import Path

this_platform = platform.machine()

platforms_requires_aux_wheels = ['armv6l', 'aarch64']
aux_wheels = {'armv6l' : [
                {'id' : '16nq1sY2-MkgNZcRhUnndTTH5WlUNM5jS', 'name' : 'depthai-0.4.1.1-cp37-cp37m-linux_armv6l.whl'}
            ],
            'aarch64' : [
                {'id' : '1CegjAQb2CD6Qagea6I34zv99Wji75LHK', 'name' : 'depthai-0.4.1.1-cp36-cp36m-linux_aarch64.whl'},
                {'id' : '1F8qfMvxv8piIzg-X2xALNvzrPFJBEqZ5', 'name' : 'PyYAML-5.3.1-cp36-cp36m-linux_aarch64.whl'}
            ]
}


# https://stackoverflow.com/a/58026969/5494277
in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_install = pip_call + ["install"]
if this_platform in platforms_requires_aux_wheels:
    pip_install = pip_install + ["--force-reinstall"]

if not in_venv:
    pip_install.append("--user")

subprocess.check_call([*pip_install, "pip", "-U"])
# temporary workaroud for issue between main and develop
subprocess.check_call([*pip_call, "uninstall", "depthai", "--yes"])
subprocess.check_call([*pip_install, "-r", "requirements.txt"])
if this_platform in platforms_requires_aux_wheels:
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
    for wheel in aux_wheels[this_platform]:
        dl_id = wheel['id']
        dl_name = wheel['name']
        wheel_dest = directory / Path(dl_name)
        download_file_from_google_drive(dl_id, wheel_dest)
        subprocess.check_call([*pip_install, str(wheel_dest)])

try:
    subprocess.check_call([*pip_install, "-r", "requirements-optional.txt"], stderr=subprocess.DEVNULL)
except subprocess.CalledProcessError as ex:
    print(f"Optional dependencies were not installed. This is not an error.")
