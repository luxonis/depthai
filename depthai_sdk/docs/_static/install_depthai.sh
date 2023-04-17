#!/bin/bash

APP_NAME="depthai"
WORKING_DIR_NAME="Luxonis"
WORKING_DIR="$HOME/$WORKING_DIR_NAME"
mkdir "$WORKING_DIR"
install_path=""
path_correct="false"

trap 'RET=$? ; echo -e >&2 "\n\x1b[31mFailed installing dependencies. Could be a bug in the installer or unsupported platform. Open a bug report over at https://github.com/luxonis/depthai - exited with status $RET at line $LINENO \x1b[0m\n" ; exit $RET' ERR

while [ "$path_correct" = "false" ]
do
  echo ""
  echo 'ENTER absolute installation path for depthai or leave empty and default path: $HOME will be used.'
  read -e install_path < /dev/tty
  echo ""

  if [ "$install_path" = "" ]; then
    echo "Using default installation path: $WORKING_DIR"
    mkdir -p "$WORKING_DIR"
  else
    echo "Using given installation path: $install_path"
    WORKING_DIR="$install_path"
  fi

  if [ -d "$WORKING_DIR" ]; then
    echo "Directory: $WORKING_DIR is OK"
    path_correct="true"
  else
    echo "Directory: $WORKING_DIR is not valid. Try again!"
  fi
done

DEPTHAI_DIR="$WORKING_DIR/$APP_NAME"
VENV_DIR="$WORKING_DIR/venv"
ENTRYPOINT_DIR="$DEPTHAI_DIR/entrypoint"

# Get Python version or find out that python 3.10 must be installed
python_executable=$(which python3)
python_chosen="false"
install_python="false"
python_version=$(python3 --version)
python_version_number=""
if [[ "$python_version" != 'Python'* ]]; then
  python_version=""
fi
echo ""

# check default python version, offer it to the user or get another one
while [ "$python_chosen" = "false" ]
do
  if [[ "$python_version" == "" ]]; then
    echo "No python version found."
    echo "Input path for python binary, version 3.8 or higher, or leave empty and python 3.10 will be installed for you."
    echo "Press any key to continue"
    read -e python_binary_path < /dev/tty
    # python not found and user wants to install python 3.10
    if [ "$python_binary_path" = "" ]; then
        install_python="true"
        python_chosen="true"
    fi
  else
    # E.g Python 3.10 -> nr_1=3, nr_2=10, for Python 3.7.5 -> nr_1=r, nr_2=7
    nr_1="${python_version:7:1}"
    nr_2=$(echo "${python_version:9:2}" | tr -d -c 0-9)
    echo "Python version: $python_version found."
    if [ "$nr_1" -gt 2 ] && [ "$nr_2" -gt 7 ]; then  # first two digits of python version greater then 3.7 -> python version 3.8 or greater is allowed.
      echo "If you want to use it for installation, press ANY key, otherwise input path to python binary."
      echo "Press any key to continue"
      read -e python_binary_path < /dev/tty
      # user wants to use already installed python whose version is high enough
      if [ "$python_binary_path" = "" ]; then
        python_chosen="true"
    fi
    else
      echo "This python version is not supported by depthai. Enter path to python binary version et least 3.8, or leave empty and python 3.10 will be installed automatically."
      echo "Press any key to continue"
      read -e python_binary_path < /dev/tty
      # python version is too low and user wants to install python 3.10
      if [ "$python_binary_path" = "" ]; then
        install_python="true"
        python_chosen="true"
      fi
    fi
  fi
  # User entered some path that should lead to python binary, save python --version output and the rest is dealt in the while loop logic.
  if [ "$python_binary_path" != "" ]; then
    python_executable="$python_binary_path"
    python_version=$($python_binary_path --version)
    if [[ "$python_version" != 'Python'* ]]; then
      python_version=""
    fi
  fi
done


write_in_file () {
  # just make sure only strings are appended which are not in there yet
  # first arg is text to write, second arg is the file path
  if ! grep -Fxq "$1" "$2"
  then
    echo "$1" >> "$2"
  fi
}

COMMENT='# Entry point for Depthai demo app, enables to run <depthai_launcher> in terminal'
BASHRC="$HOME/.bashrc"
ZSHRC="$HOME/.zshrc"
ADD_ENTRYPOINT_TO_PATH='export PATH=$PATH'":$ENTRYPOINT_DIR"

# add to .bashrc only if it is not in there already
write_in_file "$COMMENT" "$BASHRC"
write_in_file "$ADD_ENTRYPOINT_TO_PATH" "$BASHRC"

if [ -f "$ZSHRC" ]; then
  write_in_file "$COMMENT" "$ZSHRC"
  write_in_file "$ADD_ENTRYPOINT_TO_PATH" "$ZSHRC"
fi

if [[ $(uname -s) == "Darwin" ]]; then
  echo _____________________________
  echo "Calling macOS_installer.sh"
  echo _____________________________
  echo "Running macOS installer."

  echo "Installing global dependencies."
  bash -c "$(curl -fL https://docs.luxonis.com/install_dependencies.sh)"

  echo "Upgrading brew."
  brew update

  # clone depthai form git
  if [ -d "$DEPTHAI_DIR" ]; then
     echo "Demo app already downloaded. Checking out main and updating."
  else
     echo "Downloading demo app."
     git clone https://github.com/luxonis/depthai.git "$DEPTHAI_DIR"
  fi
  cd "$DEPTHAI_DIR"
  git fetch
  git checkout main
  git pull

  # install python 3.10 and python dependencies
  brew update

  if [ "$install_python" == "true" ]; then
    echo "installing python 3.10"
    brew install python@3.10
    python_executable=$(which python3.10)
  fi

  # pip does not have pyqt5 for arm
  if [[ $(uname -m) == 'arm64' ]]; then
    echo "Installing pyqt5 with homebrew."
    brew install pyqt@5
  fi

  # create python virtual environment
  echo "Creating python virtual environment in $VENV_DIR"
  echo "$python_executable"
  "$python_executable" -m venv "$VENV_DIR"
  # activate environment
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip

  # install launcher dependencies
  # only on mac silicon point PYTHONPATH to pyqt5 installation via homebrew, otherwise install pyqt5 with pip
  if [[ $(uname -m) == 'arm64' ]]; then
    if [[ ":$PYTHONPATH:" == *":/opt/homebrew/lib/python3.10/site-packages:"* ]]; then
      echo "/opt/homebrew/lib/python$nr_1.$nr_2/site-packages already in PYTHONPATH"
    else
      export "PYTHONPATH=/opt/homebrew/lib/python$nr_1.$nr_2/site-packages:"$PYTHONPATH
      echo "/opt/homebrew/lib/pythonv$nr_1.$nr_2/site-packages added to PYTHONPATH"
    fi
  else
    pip install pyqt5
  fi

  pip install packaging

elif [[ $(uname -s) == "Linux" ]]; then
  echo _____________________________
  echo "Calling linux_installer.sh"
  echo _____________________________

  echo "Updating sudo-apt."
  sudo apt-get update

  echo "Installing global dependencies."
  sudo wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash

  echo -e '\nRunning Linux installer.'

  # clone depthai form git
  if [ -d "$DEPTHAI_DIR" ]; then
     echo "Demo app already downloaded. Checking out main and updating."

  else
     echo "Downloading demo app."
     git clone https://github.com/luxonis/depthai.git "$DEPTHAI_DIR"
  fi

  cd "$DEPTHAI_DIR"
  git fetch
  git checkout main
  git pull

  # install python 3.10
  if [ "$install_python" == "true" ]; then
    echo "installing python 3.10"
  
    sudo yes "" | sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt -y install python3.10
    sudo apt -y install python3.10-venv
    python_executable=$(which python3.10)
  fi

  echo "Creating python virtual environment in $VENV_DIR"

  "$python_executable" -m venv "$VENV_DIR"

  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip

  pip install packaging
  pip install pyqt5
else
  echo "Error: Host $(uname -s) not supported."
  exit 99
fi

echo -e '\n\n:::::::::::::::: INSTALATION COMPLETE ::::::::::::::::\n'
echo -e '\nTo run demo app write <depthai_launcher> in terminal.'
echo "Press ANY KEY to finish and run the demo app..."
read -n1 key < /dev/tty
echo "STARTING DEMO APP."
python "$DEPTHAI_DIR/launcher/launcher.py" -r "$DEPTHAI_DIR"
