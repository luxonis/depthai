{
  if [[ $(pwd) != *"depthai" ]]; then
    echo "You need to run this script from the cloned depthai repository (you can clone it from here - https://github.com/luxonis/depthai)"
    exit 1
  fi

  echo "Installing required tools for the debug script..."
  sudo apt-get install util-linux usbutils lshw coreutils

  echo "Collecting HW specs..."
  echo "[lscpu]"
  sudo lscpu
  echo "[lsusb]"
  sudo lsusb
  echo "[uname -a]"
  sudo uname -a
  echo "[lshw -short]"
  sudo lshw -short
  echo "[lsblk]"
  sudo lsblk

  echo
  echo "Collecting installed software..."
  echo "[apt list --installed]"
  sudo apt list --installed
  echo "[pip freeze]"
  python3 -m pip freeze

  echo
  echo "Checking the environment..."
  echo "[printenv]"
  printenv

  echo
  echo "Checking the versions..."
  echo "[python3 --version]"
  python3 --version
  echo "[python3 -m pip --version]"
  python3 -m pip --version
  echo "[git log -n 1]"
  git log -n 1
  echo "[git branch]"
  git branch
  echo "[git submodule status]"
  git submodule status
} > log_system_information.txt
