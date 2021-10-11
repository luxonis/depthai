#/usr/bin/env python3

# Launcher for depthai_demo.py which provides updating capabilities

# Standard imports
import os, sys, subprocess, time, threading, argparse
# Import splash screen
import pyqt5_splash_screen
# Import version parser
from packaging import version
# PyQt5
from PyQt5 import QtCore, QtGui, QtWidgets

# Constants
SCRIPT_DIRECTORY=os.path.abspath(os.path.dirname(__file__))
DEPTHAI_DEMO_SCRIPT='depthai_demo.py'
DEPTHAI_INSTALL_REQUIREMENTS_SCRIPT='install_requirements.py'
DEFAULT_GIT_PATH='git'
DEPTHAI_REPOSITORY_NAME = 'depthai'
DEPTHAI_REMOTE_REPOSITORY_URL = 'https://github.com/luxonis/depthai.git'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repo', help='Path to DepthAI Git repository', default=f'{SCRIPT_DIRECTORY}/{DEPTHAI_REPOSITORY_NAME}')
parser.add_argument('-g', '--git', help='Path to Git executable. Default \'git\'', default=DEFAULT_GIT_PATH)
args = parser.parse_args()

pathToDepthaiRepository = args.repo
gitExecutable = args.git

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

qApp = QtWidgets.QApplication(['DepthAI Launcher'])
# Set style
#print(PyQt5.QtWidgets.QStyleFactory.keys())
#qApp.setStyle('Fusion')


splashScreen = pyqt5_splash_screen.SplashScreen('splash2.png')

def closeSplash():
    splashScreen.hide()

class Worker(QtCore.QThread):
    signalUpdateQuestion = QtCore.pyqtSignal(str, str)
    sigInfo = QtCore.pyqtSignal(str, str)
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

    def __init__(self, parent = None):
        QtCore.QThread.__init__(self, parent)
        self.signalUpdateQuestion[str, str].connect(self.updateQuestion, QtCore.Qt.BlockingQueuedConnection)
        self.sigInfo[str, str].connect(self.showInformation, QtCore.Qt.BlockingQueuedConnection)
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

            # Check if repository exists
            try:

                subprocess.run([gitExecutable, 'status'], cwd=pathToDepthaiRepository)

                if os.path.isdir(pathToDepthaiRepository) and subprocess.run([gitExecutable, 'status'], cwd=pathToDepthaiRepository).returncode == 0:
                    pass
                else:
                    # DepthAI repo not available, clone first
                    splashScreen.updateSplashMessage('Loading DepthAI Repository ...')
                    splashScreen.enableHeartbeat(True)
                    # Repository doesn't exists, clone first
                    subprocess.check_call([gitExecutable, 'clone', DEPTHAI_REMOTE_REPOSITORY_URL, DEPTHAI_REPOSITORY_NAME], cwd=SCRIPT_DIRECTORY)

                # Fetch changes
                subprocess.check_call([gitExecutable, 'fetch'], cwd=pathToDepthaiRepository)

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
                if len(availableDepthAIVersions) > 0:
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
                    # See if its DepthAI version (if not, then suggest to update)
                    if len(tag.split('-sdk')) == 1:
                        splitTag = tag.split('-')[0]
                        splitTag = splitTag.split('v')
                        currentTag = splitTag[len(splitTag) - 1]
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

                        if self.shouldUpdate == True:
                            # DepthAI repo not available, clone first
                            splashScreen.updateSplashMessage('Updating DepthAI Repository ...')
                            splashScreen.enableHeartbeat(True)
                            subprocess.run([gitExecutable, 'checkout', newVersionTag], cwd=pathToDepthaiRepository, check=True)

                            # present message of installing dependencies
                            splashScreen.updateSplashMessage('Loading DepthAI Dependencies ...')
                            splashScreen.enableHeartbeat(True)

                            # Install requirements for depthai_demo.py
                            subprocess.run([sys.executable, f'{pathToDepthaiRepository}/{DEPTHAI_INSTALL_REQUIREMENTS_SCRIPT}'], cwd=pathToDepthaiRepository)

            except subprocess.CalledProcessError as ex:
                # TODO(themarpe) - issue information box that Git isn't available
                title = 'Git Error'
                message = f'Git produced the following error: {ex}'
                print(f'Message Box ({title}): {message}')
                self.sigInfo.emit(title, message)
                raise Exception('Git Error')
            except FileNotFoundError as ex:
                # TODO(themarpe) - issue information box that Git isn't available
                title = 'No Git Available'
                message = 'Git cannot be found in the path. Make sure Git is installed and added to the path, then try again'
                print(f'Message Box ({title}): {message}')
                self.sigInfo.emit(title, message)
                raise Exception('No Git Found')

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

                # Retry if failed by an ModuleNotFoundError, by installing the requirements
                if ret.returncode != 0 and ('ModuleNotFoundError' in str(ret.stderr) or 'Version mismatch' in str(ret.stderr)):
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
sys.exit(qApp.exec_())
