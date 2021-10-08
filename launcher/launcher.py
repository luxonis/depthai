#/usr/bin/env python3

# Launcher for depthai_demo.py which provides updating capabilities

# Standard imports
import os, sys, subprocess, time, threading

# Constants
SCRIPT_DIRECTORY=os.path.abspath(os.path.dirname(__file__))
DEPTHAI_DEMO_SCRIPT='depthai_demo.py'
DEPTHAI_INSTALL_REQUIREMENTS_SCRIPT='install_requirements.py'

# Create a logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f'{SCRIPT_DIRECTORY}/log.dat', 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
# Write both stdout and stderr to log files
# Note - doesn't work for subprocesses.
# Do proper fd dup for that case
logger = Logger()
sys.stdout = logger
sys.stderr = logger

# PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets

qApp = QtWidgets.QApplication(['DepthAI Launcher'])

# Import splash screen
import pyqt5_splash_screen

splashScreen = pyqt5_splash_screen.SplashScreen('splash2.png')


def workingThread():

    def closeSplash():
        splashScreen.hide()

    # # Create splash screen with splash2.png image
    # splashProcess = subprocess.Popen([sys.executable, f'{SCRIPT_DIRECTORY}/splash_screen.py', 'splash2.png', '0'], stdin=subprocess.PIPE, text=True)
    # splashClosed = False
    # def closeSplash():
    #     global splashClosed
    #     if not splashClosed:
    #         splashClosed = True
    #         splashProcess.terminate()
    #         #splashProcess.communicate(input='a', timeout=1.0)

    # Defaults
    depthaiRepositoryName = 'depthai'
    pathToDepthaiRemoteRepository = 'https://github.com/luxonis/depthai.git'
    pathToDepthaiRepository = f'{SCRIPT_DIRECTORY}/{depthaiRepositoryName}'
    shouldUpdate = True

    # If path to repository is specified, use that instead
    # TODO(themarpe) - Expand possible arguments
    if len(sys.argv) > 1:
        pathToDepthaiRepository = sys.argv[1]

    # Check if repository exists
    from dulwich.repo import Repo
    from dulwich import porcelain
    import dulwich

    depthaiRepo = None
    try:
        depthaiRepo = Repo(pathToDepthaiRepository)
    except dulwich.errors.NotGitRepository as ex:
        launcher.updateSplashMessage('DepthAI repository')
        launcher.enableHeartbeat(True)

        # Repository doesn't exists, clone first
        depthaiRepo = porcelain.clone(pathToDepthaiRemoteRepository, depthaiRepositoryName)

    # # Check if an update is available
    # # TODO(themarpe)
    # from dulwich.repo import Repo
    # import dulwich.porcelain
    #
    # r = Repo(pathToDepthaiRepository)
    # # Fetch tags first
    # dulwich.porcelain.fetch(r)
    # # Get current tag
    # currentTag = dulwich.porcelain.describe(r)
    #
    # tags = r.refs.as_dict("refs/tags".encode())
    #
    # print(f'Current tag: {dulwich.porcelain.describe(r)}')
    #
    # ver = semver.VersionInfo.parse(currentTag)
    # print(ver)
    # print(tags)
    newVersionAvailable = True

    # If a new version is available, ask to update
    if newVersionAvailable == True:
        # Try asking user whether to update
        # import semver
        # Update by default
        print("Message Box in Console")
        ret = QtWidgets.QMessageBox.question(None, "Update Available", "Version 2.3.1 is available.\nCurrent version is 2.2.0.\nUpdate?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        print(f'Should update? {shouldUpdate}')
        if ret == QtWidgets.QMessageBox.Yes:
            shouldUpdate = True

        if shouldUpdate == True:
            pass
            #TODO (themarpe) - update by fetch & checkout of depthai repo

    # Set to quit splash screen a little after subprocess is ran
    skipSplashQuitFirstTime = False
    def removeSplash():
        time.sleep(2.5)
        if not skipSplashQuitFirstTime:
            closeSplash()
    quitThread = threading.Thread(target=removeSplash)
    quitThread.start()

    # All ready, run the depthai_demo.py as a separate process
    ret = subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_DEMO_SCRIPT}'], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)

    # Print out stderr first
    sys.stderr.write(ret.stderr.decode())

    # Retry if failed by an ModuleNotFoundError, by installing the requirements
    if ret.returncode != 0 and 'ModuleNotFoundError' in str(ret.stderr):
        skipSplashQuitFirstTime = True
        print(f'ModuleNotFoundError raised. Retrying by installing requirements first and restarting demo.')

        # present message of installing dependencies
        splashScreen.updateSplashMessage('Loading DepthAI Dependencies ...')
        splashScreen.enableHeartbeat(True)

        # Install requirements for depthai_demo.py
        subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_INSTALL_REQUIREMENTS_SCRIPT}'], cwd=pathToDepthaiRepository)

        # Remove message and animation
        splashScreen.updateSplashMessage('')
        splashScreen.enableHeartbeat(False)

        quitThread.join()
        skipSplashQuitFirstTime = False
        quitThread = threading.Thread(target=removeSplash)
        quitThread.start()

        # All ready, run the depthai_demo.py as a separate process
        subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_DEMO_SCRIPT}'], cwd=pathToDepthaiRepository)

    # At the end quit anyway
    closeSplash()
    quitThread.join()
    splashScreen.close()
    qApp.exit()

threading.Thread(target=workingThread).start()
sys.exit(qApp.exec_())
