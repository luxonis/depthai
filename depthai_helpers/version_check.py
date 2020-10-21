import depthai
from pathlib import Path


def get_version_from_requirements():
    with Path('requirements.txt').open() as f:
        datafile = f.readlines()
    for line in datafile:
        if 'depthai' in line:
            #not commented out
            if not line.startswith('#'):
                try:
                    version = line.split('==')[1]
                    #remove any whitespace
                    version = version.strip()
                except:
                    version = None
                return version
    return None

def check_depthai_version():
    version_required = get_version_from_requirements()
    if version_required is not None:
        print('Depthai version required:  ', version_required)
        if depthai.__version__.endswith('+dev'):
            print('Depthai development version found, skipping check.')
        elif version_required != depthai.__version__:
            raise ValueError(f"\033[1;5;31mVersion mismatch \033[0m\033[91m between installed depthai lib and the required one by the script.\033[0m \n\
                Required:  {version_required}\n\
                Installed: {depthai.__version__}\n\
                \033[91mRun: ./install_requirements.sh \033[0m")