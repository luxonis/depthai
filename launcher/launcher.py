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

qApp = QtWidgets.QApplication(['DepthAI Launcher'])
# Set style
#print(PyQt5.QtWidgets.QStyleFactory.keys())
#qApp.setStyle('Fusion')
# Set default Window icon
qApp.setWindowIcon(QtGui.QIcon(str(SCRIPT_DIRECTORY/'splash2.png')))
# Create splash screen
splashScreen = SplashScreen(str(SCRIPT_DIRECTORY/'splash2.png'))

def closeSplash():
    splashScreen.hide()

class Worker(QtCore.QThread):
    signalUpdateQuestion = QtCore.pyqtSignal(str, str)
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

                    # Get all available versions
                    availableDepthAIVersions = []
                    proc = subprocess.Popen([gitExecutable, 'tag', '-l'], cwd=pathToDepthaiRepository, stdout=subprocess.PIPE)
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        # Check that the tag refers to DepthAI demo and not SDK
                        tag = line.rstrip().decode()
                        # Check that tag is actually a version
                        if type(version.parse(tag)) is version.Version:
                            availableDepthAIVersions.append(tag)
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

                    # Check current tag
                    ret = subprocess.run([gitExecutable, 'describe', '--tags'], cwd=pathToDepthaiRepository, stdout=subprocess.PIPE, check=True)
                    tag = ret.stdout.decode()
                    # See if its DepthAI version tag (if not, then suggest to update)
                    if len(tag.split('-')) == 1:
                        currentVersion = 'Unknown'
                        if type(version.parse(tag)) is version.Version:
                            print(f'Current tag: {tag}, ver: {str(version.parse(tag))}')
                            currentVersion = str(version.parse(tag))

                            # Check if latest version is newer than current
                            if version.parse(newVersionTag) > version.parse(tag):
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

            try:
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

        except Exception as ex:
            # Catch all for any kind of an error
            print(f'Unknown error occured ({ex}), exiting...')
        finally:
            # At the end quit anyway
            closeSplash()
            splashScreen.close()
            qApp.exit()

qApp.worker = Worker()
qApp.worker.start()
sys.exit(qApp.exec())
