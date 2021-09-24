import depthai
from pathlib import Path


def getVersionFromRequirements():
    requirementsPath = Path(__file__).parent / Path('../requirements.txt')
    with requirementsPath.open() as f:
        datafile = f.readlines()
    for line in datafile:
        if 'depthai' in line:
            #not commented out
            if not line.startswith('#'):
                try:
                    version = line.split('==')[1]
                    version = version.split(';')[0]
                    #remove any whitespace
                    version = version.strip()
                except:
                    version = None
                return version
    return None

def checkDepthaiVersion():
    versionRequired = getVersionFromRequirements()
    if versionRequired is not None:
        if depthai.__version__.endswith('+dev'):
            print('Depthai development version found, skipping check.')
        elif versionRequired != depthai.__version__:
            raise SystemExit(f"\033[1;5;31mVersion mismatch\033[0m\033[91m between installed depthai lib and the required one by the script.\033[0m \n\
                Required:  {versionRequired}\n\
                Installed: {depthai.__version__}\n\
                \033[91mRun: python3 install_requirements.py \033[0m")
