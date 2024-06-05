#/usr/bin/env python3

# Launcher for depthai_demo.py which provides updating capabilities

# Standard imports
import os, sys, subprocess, time, threading, argparse, datetime

import re
from pathlib import Path
# Import splash screen
from splash_screen import SplashScreen
# Import version parser
from packaging import version
# PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets

from choose_app_dialog import ChooseAppDialog

# Constants
SCRIPT_DIRECTORY=Path(os.path.abspath(os.path.dirname(__file__)))
DEPTHAI_DEMO_SCRIPT='depthai_demo.py'
DEPTHAI_INSTALL_REQUIREMENTS_SCRIPT='install_requirements.py'
DEFAULT_GIT_PATH='git'
DEPTHAI_REPOSITORY_NAME = 'depthai'
DEPTHAI_REMOTE_REPOSITORY_URL = 'https://github.com/luxonis/depthai.git'
LOG_FILE_PATH=Path(SCRIPT_DIRECTORY/'log.dat')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repo', help='Path to DepthAI Git repository', default=SCRIPT_DIRECTORY/'..')
parser.add_argument('-g', '--git', help='Path to Git executable. Default: %(default)s', default=DEFAULT_GIT_PATH)
parser.add_argument('--disable-git', help='Disable git requirement and updating capability', default=False, action='store_true')
args = parser.parse_args()

pathToDepthaiRepository = args.repo
gitExecutable = args.git
if args.disable_git:
    gitExecutable = ''

# Create a logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(LOG_FILE_PATH, 'a')
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

print(f'========= Starting: Launcher ({datetime.datetime.now()}) =========')
print(f"System information:")
print(f" - sys.prefix = {sys.prefix}")
print(f" - sys.base_prefix = {sys.base_prefix}")
print(f" - Env 'PYTHONPATH': {os.getenv('PYTHONPATH')}")
print(f" - Executable: {sys.executable}\n")

qApp = QtWidgets.QApplication(['DepthAI Launcher'])
# Set style
#print(PyQt5.QtWidgets.QStyleFactory.keys())
#qApp.setStyle('Fusion')
# Set default Window icon
qApp.setWindowIcon(QtGui.QIcon(str(SCRIPT_DIRECTORY/'splash2.png')))
# Create splash screen
splashScreen = SplashScreen(str(SCRIPT_DIRECTORY/'splash2.png'))


class Worker(QtCore.QThread):
    signalUpdateQuestion = QtCore.pyqtSignal(str, str)
    signalChooseApp = QtCore.pyqtSignal()
    signalCloseSplash = QtCore.pyqtSignal()
    sigInfo = QtCore.pyqtSignal(str, str)
    sigCritical = QtCore.pyqtSignal(str, str)
    sigWarning = QtCore.pyqtSignal(str, str)
    # Should update if a new version is available?
    shouldUpdate = True

    @QtCore.pyqtSlot(str,str)
    def updateQuestion(self, title, message):
        ret = QtWidgets.QMessageBox.question(splashScreen, title, message, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.Yes)
        if ret == QtWidgets.QMessageBox.Yes:
            self.shouldUpdate = True
            return True
        else:
            self.shouldUpdate = False
            return False
        
    @QtCore.pyqtSlot()
    def chooseApp(self) -> None:
        """
        Until Depthai Viewer is in beta, allow the user to choose between running the demo or the viewer.
        """
        # If the dialog is rejected, the user has clicked exit - so we exit
        dialog = ChooseAppDialog(splashScreen)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.viewerChosen = dialog.viewerChosen
        else:
            raise RuntimeError("User cancelled app choice dialog")
        
    @QtCore.pyqtSlot()
    def closeSplash(self):
        splashScreen.close()

    @QtCore.pyqtSlot(str,str)
    def showInformation(self, title, message):
        QtWidgets.QMessageBox.information(splashScreen, title, message)

    @QtCore.pyqtSlot(str,str)
    def showWarning(self, title, message):
        QtWidgets.QMessageBox.warning(splashScreen, title, message)

    @QtCore.pyqtSlot(str,str)
    def showCritical(self, title, message):
        QtWidgets.QMessageBox.critical(splashScreen, title, message)

    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.signalUpdateQuestion[str, str].connect(self.updateQuestion, QtCore.Qt.BlockingQueuedConnection)
        # Commented out for running the viewer directly, without the option of choosing demo
        #self.signalChooseApp.connect(self.chooseApp, QtCore.Qt.BlockingQueuedConnection)
        self.signalCloseSplash.connect(self.closeSplash, QtCore.Qt.BlockingQueuedConnection)
        self.sigInfo[str, str].connect(self.showInformation, QtCore.Qt.BlockingQueuedConnection)
        self.sigCritical[str, str].connect(self.showCritical, QtCore.Qt.BlockingQueuedConnection)
        self.sigWarning[str, str].connect(self.showWarning, QtCore.Qt.BlockingQueuedConnection)
    def __del__(self):
        self.exiting = True
        try:
            self.wait()
        except:
            pass

    def run(self):

        try:

            # New version available?
            newVersionAvailable = False
            # Current version name
            currentVersion = 'Unknown'
            newVersion = 'Unknown'
            newVersionTag = 'vUnknown'
            lastCall = ''

            try:

                # Check if 'disable git' option was specified
                if gitExecutable != '':


                    # Check if repository exists
                    if os.path.isdir(pathToDepthaiRepository) and subprocess.run([gitExecutable, 'status'], cwd=pathToDepthaiRepository).returncode == 0:
                        pass
                    else:
                        # DepthAI repo not available, clone first
                        splashScreen.updateSplashMessage('Cloning DepthAI Repository ...')
                        splashScreen.enableHeartbeat(True)
                        # Repository doesn't exists, clone first
                        subprocess.check_call([gitExecutable, 'clone', DEPTHAI_REMOTE_REPOSITORY_URL, DEPTHAI_REPOSITORY_NAME], cwd=SCRIPT_DIRECTORY)

                    # Fetch changes
                    # Save error of an possible no internet connection scenario
                    lastCall = subprocess.run([gitExecutable, 'fetch'], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)
                    lastCall.check_returncode()

                    # Get current commit
                    ret = subprocess.run([gitExecutable, 'rev-parse', 'HEAD'], cwd=pathToDepthaiRepository, stdout=subprocess.PIPE, check=True)
                    currentCommit = ret.stdout.decode().strip()
                    print(f'Current commit: {currentCommit}')

                    # Tags associated with current commit
                    currentTag = None

                    # Get all available tags/versions
                    availableDepthAIVersions = []
                    proc = subprocess.Popen([gitExecutable, 'show-ref', '--tags', '-d'], cwd=pathToDepthaiRepository, stdout=subprocess.PIPE)
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        # Parse commit and corresponding tags
                        commitTag = line.rstrip().decode()
                        tag = ''
                        commit = ''
                        try:
                            commit = commitTag.split(' ')[0].strip()
                            tag = commitTag.split(' ')[1].split('refs/tags/')[1].split("^{}")[0].strip()
                        except Exception as ex:
                            print(f"Couldn't parse commit&tag line: {ex}")

                        # Check that tag is actually a version
                        # Check that the tag belongs to depthai demo and not SDK or others and is valid
                        if len(tag.split('-')) == 1 and type(version.parse(tag)) is version.Version:
                            availableDepthAIVersions.append(tag)
                            # Also note down the current depthai demo tag
                            if currentCommit == commit:
                                currentTag = tag

                    # Print available tags/versions
                    print(f'Current tag: {currentTag}')
                    print(f'Available DepthAI versions: {availableDepthAIVersions}')

                    # If any available versions
                    if len(availableDepthAIVersions) == 0:
                        raise RuntimeError('No available depthai versions found')

                    # Assuming versions are available
                    # Get latest version
                    newVersionTag = availableDepthAIVersions[0]
                    newVersion = str(version.parse(newVersionTag))
                    for ver in availableDepthAIVersions:
                        if version.parse(ver) > version.parse(newVersionTag):
                            newVersionTag = ver
                            newVersion = str(version.parse(ver))

                    # See if its DepthAI version tag (if not, then suggest to update)
                    if currentTag is not None:
                        currentVersion = 'Unknown'
                        if type(version.parse(currentTag)) is version.Version:
                            print(f'Current tag: {currentTag}, ver: {str(version.parse(currentTag))}')
                            currentVersion = str(version.parse(currentTag))

                            # Check if latest version is newer than current
                            if version.parse(newVersionTag) > version.parse(currentTag):
                                newVersionAvailable = True
                            else:
                                newVersionAvailable = False

                        else:
                            newVersionAvailable = True
                    else:
                        newVersionAvailable = True

                    # If a new version is available, ask to update
                    if newVersionAvailable == True:
                        # Ask user whether to update
                        # Update by default
                        title = 'Update Available'
                        message = f'Version {newVersion} is available.\nCurrent version is {currentVersion}\nUpdate?'
                        print(f'Message Box ({title}): {message}')
                        self.signalUpdateQuestion.emit(title, message)

                        print(f'Should update? {self.shouldUpdate}')
                        didUpdate = False
                        if self.shouldUpdate == True:
                            # DepthAI repo not available, clone first
                            splashScreen.updateSplashMessage('Updating DepthAI Repository ...')
                            splashScreen.enableHeartbeat(True)
                            lastCall = subprocess.run([gitExecutable, 'status', '--porcelain'], cwd=pathToDepthaiRepository, stdout=subprocess.PIPE)
                            filesToRemove = lastCall.stdout.decode()
                            lastCall = subprocess.run([gitExecutable, 'checkout', '--recurse-submodules', newVersionTag], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)
                            if lastCall.returncode != 0 or filesToRemove != "":
                                # Uncommited changes - redo with a prompt to force
                                # Or unclean working directory
                                errMessage = lastCall.stderr.decode()
                                title = 'Force Update'
                                message = f'DepthAI Repository has changes. Do you want to override the changes?'
                                if lastCall.returncode != 0:
                                    message = f'{message}\n{errMessage}'
                                if filesToRemove != "":
                                    message = f'{message}\nWould also remove:\n{filesToRemove}'
                                print(f'Message Box ({title}): {message}')
                                self.signalUpdateQuestion.emit(title, message)

                                if self.shouldUpdate == True:
                                    lastCall = subprocess.run([gitExecutable, 'checkout', '--recurse-submodules', '-f', newVersionTag], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)
                                    checkoutSuccess = lastCall.returncode == 0
                                    lastCall = subprocess.run([gitExecutable, 'clean', '-fd'], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)
                                    cleanSuccess = lastCall.returncode == 0
                                    if checkoutSuccess == False or cleanSuccess == False:
                                        # Stop animation
                                        splashScreen.updateSplashMessage('')
                                        splashScreen.enableHeartbeat(False)
                                        # Couldn't update. Issue a warning
                                        errMessage = lastCall.stderr.decode()
                                        title = 'Update Aborted'
                                        message = f'DepthAI Repository could not be updated.\n{errMessage}'
                                        print(f'Message Box ({title}): {message}')
                                        self.sigWarning.emit(title, message)
                                    else:
                                        didUpdate = True
                            else:
                                didUpdate = True

                        if didUpdate:
                            currentArgs = []
                            if len(sys.argv) >= 1:
                                currentArgs = sys.argv[1:]
                            arguments = [sys.executable, f'{pathToDepthaiRepository}/launcher/launcher.py'] + currentArgs
                            # Run updated launcher
                            print('Updated, running new launcher - command: ' + str(arguments))
                            subprocess.Popen(arguments, cwd=pathToDepthaiRepository)
                            # Exit current launcher
                            raise Exception('Shutting down and starting updated launcher')

            except subprocess.CalledProcessError as ex:
                errMessage = lastCall.stderr.decode()
                title = 'Git Error'
                message = f'Git produced the following error: {ex}\nOutput: {errMessage}'
                print(f'Message Box ({title}): {message}')
                #self.sigInfo.emit(title, message)
                #raise Exception('Git Error')
            except FileNotFoundError as ex:
                # Stop animation
                splashScreen.updateSplashMessage('')
                splashScreen.enableHeartbeat(False)
                title = 'No Git Available'
                message = 'Git cannot be found in the path. Make sure Git is installed and added to the path, then try again'
                print(f'Message Box ({title}): {message}')
                # TODO(themarpe) - could be made optional, if the following raise and message
                self.sigCritical.emit(title, message)
                raise Exception('No Git Found')
            except RuntimeError as ex:
                # Stop animation
                splashScreen.updateSplashMessage('')
                splashScreen.enableHeartbeat(False)
                title = 'No DepthAI Versions Found'
                message = "Couldn't find any available DepthAI versions. Continuing with existing version. Please report to developers."
                print(f'Message Box ({title}): {message}')
                # TODO(themarpe) - could be made optional, if the following raise and message
                self.sigWarning.emit(title, message)
            '''
            try:
                self.signalChooseApp.emit()
                # Set to quit splash screen a little after subprocess is ran
                skipSplashQuitFirstTime = False
                def removeSplash():
                    time.sleep(2.5)
                    if not skipSplashQuitFirstTime:
                        self.signalCloseSplash.emit()
                quitThread = threading.Thread(target=removeSplash)
                quitThread.start()
                if self.viewerChosen:
                    print("Depthai Viewer chosen, checking if depthai-viewer is installed.")
                    # Check if depthai-viewer is installed
                    is_viewer_installed_cmd = [sys.executable, "-m", "pip", "show", "depthai-viewer"]
                    viewer_available_ret = subprocess.run(is_viewer_installed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if viewer_available_ret.returncode != 0:
                        splashScreen.updateSplashMessage('Installing Depthai Viewer ...')
                        splashScreen.enableHeartbeat(True)
                        print("Depthai Viewer not installed, installing...")
                        # Depthai Viewer isn't installed, install it
                        # First upgrade pip
                        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)
                        # Install depthai-viewer - Don't check, it can error out because of dependency conflicts but still install successfully
                        subprocess.run([sys.executable, "-m", "pip", "install", "depthai-viewer"]) 
                        # Check again if depthai-viewer is installed
                        viewer_available_ret = subprocess.run(is_viewer_installed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if viewer_available_ret.returncode != 0:
                            raise RuntimeError("Depthai Viewer failed to install.")
                    splashScreen.updateSplashMessage('')
                    splashScreen.enableHeartbeat(False)

                    viewer_version = version.parse(viewer_available_ret.stdout.decode().splitlines()[1].split(" ")[1].strip())
                    print(f"Installed Depthai Viewer version: {viewer_version}")
                    # Get latest depthai-viewer version
                    latest_ret = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "depthai-viewer"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if latest_ret.returncode != 0:
                        raise RuntimeError("Couldn't get latest depthai-viewer version.")
                    latest_viewer_version = version.parse(latest_ret.stdout.decode().split("LATEST:")[1].strip())
                    print(f"Latest Depthai Viewer version: {latest_viewer_version}")
                    if latest_viewer_version > viewer_version:
                        # Update is available, ask user if they want to update
                        title = 'DepthAI Viewer update available'
                        message = f'Version {str(latest_viewer_version)} of depthai-viewer is available, current version {str(viewer_version)}. Would you like to update?'
                        self.signalUpdateQuestion.emit(title, message)
                        if self.shouldUpdate:
                            splashScreen.updateSplashMessage(f'Updating Depthai Viewer to version {latest_viewer_version} ...')
                            splashScreen.enableHeartbeat(True)
                            # Update depthai-viewer
                            subprocess.run([sys.executable, "-m", "pip", "install", "-U", "depthai-viewer"])
                            # Test again to see if viewer is installed and updated
                            viewer_available_ret = subprocess.run(is_viewer_installed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            if viewer_available_ret.returncode != 0:
                                raise RuntimeError(f"Installing version {latest_viewer_version} failed.")
                            viewer_version = version.parse(viewer_available_ret.stdout.decode().splitlines()[1].split(" ")[1].strip())
                            if latest_viewer_version > viewer_version:
                                raise RuntimeError("Depthai Viewer failed to update.")
                            splashScreen.updateSplashMessage('')
                            splashScreen.enableHeartbeat(False)

                    # All ready, run the depthai-viewer as a seperate process
                    ret = subprocess.run([sys.executable, "-m", "depthai_viewer"])
                else:    
                    # All ready, run the depthai_demo.py as a separate process
                    ret = subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_DEMO_SCRIPT}'], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)

                    # Print out stderr first
                    sys.stderr.write(ret.stderr.decode())

                    print(f'DepthAI Demo ret code: {ret.returncode}')
                    # Install dependencies if demo signaled missing dependencies
                    if ret.returncode == 42:
                        skipSplashQuitFirstTime = True
                        print(f'Dependency issue raised. Retrying by installing requirements and restarting demo.')

                        # present message of installing dependencies
                        splashScreen.updateSplashMessage('Installing DepthAI Requirements ...')
                        splashScreen.enableHeartbeat(True)

                        # Install requirements for depthai_demo.py
                        MAX_RETRY_COUNT = 3
                        installReqCall = None
                        for retry in range(0, MAX_RETRY_COUNT):
                            installReqCall = subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_INSTALL_REQUIREMENTS_SCRIPT}'], cwd=pathToDepthaiRepository, stderr=subprocess.PIPE)
                            if installReqCall.returncode == 0:
                                break
                        if installReqCall.returncode != 0:
                            # Some error happened. Notify user
                            title = 'Error Installing DepthAI Requirements'
                            message = f"Couldn't install DepthAI requirements. Check internet connection and try again. Log available at: {LOG_FILE_PATH}"
                            print(f'Message Box ({title}): {message}')
                            print(f'Install dependencies call failed with return code: {installReqCall.returncode}, message: {installReqCall.stderr.decode()}')
                            self.sigCritical.emit(title, message)
                            raise Exception(title)

                        # Remove message and animation
                        splashScreen.updateSplashMessage('')
                        splashScreen.enableHeartbeat(False)

                        quitThread.join()
                        skipSplashQuitFirstTime = False
                        quitThread = threading.Thread(target=removeSplash)
                        quitThread.start()

                        # All ready, run the depthai_demo.py as a separate process
                        subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_DEMO_SCRIPT}'], cwd=pathToDepthaiRepository)
            except:
                pass
            finally:
                quitThread.join()
            '''
            ### Replaced above commented code with below code unitl ###
            # No option of choosing demo, running the viewer directly
            try:
                # Set to quit splash screen a little after subprocess is ran
                def removeSplash():
                    time.sleep(2.5)
                    self.signalCloseSplash.emit()
                quitThread = threading.Thread(target=removeSplash)
                quitThread.start()

                print("Launching DepthAI Viewer.")
                # Check if depthai-viewer is installed
                is_viewer_installed_cmd = [sys.executable, "-m", "pip", "show", "depthai-viewer"]
                viewer_available_ret = subprocess.run(is_viewer_installed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if viewer_available_ret.returncode != 0:
                    splashScreen.updateSplashMessage('Installing DepthAI Viewer ...')
                    splashScreen.enableHeartbeat(True)
                    print("DepthAI Viewer not installed, installing...")
                    # DepthAI Viewer isn't installed, install it
                    # First upgrade pip
                    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "pip"], check=True)
                    # Install depthai-viewer - Don't check, it can error out because of dependency conflicts but still install successfully
                    subprocess.run([sys.executable, "-m", "pip", "install", "depthai-viewer"]) 
                    # Check again if depthai-viewer is installed
                    viewer_available_ret = subprocess.run(is_viewer_installed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if viewer_available_ret.returncode != 0:
                        raise RuntimeError("DepthAI Viewer failed to install.")
                splashScreen.updateSplashMessage('')
                splashScreen.enableHeartbeat(False)

                viewer_version = version.parse(viewer_available_ret.stdout.decode().splitlines()[1].split(" ")[1].strip())
                print(f"Installed DepthAI Viewer version: {viewer_version}")
                # Get latest depthai-viewer version
                latest_ret = subprocess.run([sys.executable, "-m", "pip", "index", "versions", "depthai-viewer"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if latest_ret.returncode != 0:
                    raise RuntimeError("Couldn't get latest depthai-viewer version.")
                latest_viewer_version = version.parse(latest_ret.stdout.decode().split("LATEST:")[1].strip())
                print(f"Latest DepthAI Viewer version: {latest_viewer_version}")
                if latest_viewer_version > viewer_version:
                    # Update is available, ask user if they want to update
                    title = 'DepthAI Viewer update available'
                    message = f'Version {str(latest_viewer_version)} of depthai-viewer is available, current version {str(viewer_version)}. Would you like to update?'
                    self.signalUpdateQuestion.emit(title, message)
                    if self.shouldUpdate:
                        splashScreen.updateSplashMessage(f'Updating DepthAI Viewer to version {latest_viewer_version} ...')
                        splashScreen.enableHeartbeat(True)
                        # Update depthai-viewer
                        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "depthai-viewer"])
                        # Test again to see if viewer is installed and updated
                        viewer_available_ret = subprocess.run(is_viewer_installed_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if viewer_available_ret.returncode != 0:
                            raise RuntimeError(f"Installing version {latest_viewer_version} failed.")
                        viewer_version = version.parse(viewer_available_ret.stdout.decode().splitlines()[1].split(" ")[1].strip())
                        if latest_viewer_version > viewer_version:
                            raise RuntimeError("DepthAI Viewer failed to update.")
                        splashScreen.updateSplashMessage('')
                        splashScreen.enableHeartbeat(False)

                # All ready, run the depthai-viewer as a seperate process
                ret = subprocess.run([sys.executable, "-m", "depthai_viewer"])
            except Exception as ex:
                print(f'Exception: {ex}')
                title = 'Exception'
                message = f'Unable to start the DepthAI Viewer.\nException: {ex}'
                print(f'Message Box ({title}): {message}')
                self.sigCritical.emit(title, message)
            finally:
                quitThread.join()
            ###
        except Exception as ex:
            # Catch all for any kind of an error
            print(f'Unknown error occured ({ex}), exiting...')
        finally:
            # At the end quit anyway
            self.signalCloseSplash.emit()
            qApp.exit()

qApp.worker = Worker()
qApp.worker.start()
sys.exit(qApp.exec())
