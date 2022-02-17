# Constants
# Build directory
$BUILD_DIR = "$PSScriptRoot\build"
# WinPython embedded Python3.9
$EMBEDDED_PYTHON="https://github.com/winpython/winpython/releases/download/4.3.20210620/Winpython64-3.9.5.0dot.exe"
$PORTABLE_GIT="https://github.com/git-for-windows/git/releases/download/v2.33.0.windows.2/PortableGit-2.33.0.2-64-bit.7z.exe"
# Save the current location and switch to this script's directory.
# Note: This shouldn't fail; if it did, it would indicate a
#       serious system-wide problem.
$prevPwd = $PWD; Set-Location -ErrorAction Stop -LiteralPath $PSScriptRoot

try {
    
    # Create build directory
    md -Force "$BUILD_DIR"
    # Download WinPython
    $ProgressPreference = 'SilentlyContinue'    # Subsequent calls do not display UI.
    Invoke-WebRequest "$EMBEDDED_PYTHON" -OutFile "$BUILD_DIR\winpython39.exe"
    $ProgressPreference = 'Continue'            # Subsequent calls do display UI.
    #wget $EMBEDDED_PYTHON -O "$BUILD_DIR\winpython39.exe"
    # Extract the contents
    Start-Process "$BUILD_DIR\winpython39.exe" -NoNewWindow -Wait -ArgumentList "-o`"$BUILD_DIR`" -y" 
    
    # Download Portable Git
    $ProgressPreference = 'SilentlyContinue'    # Subsequent calls do not display UI.
    Invoke-WebRequest "$PORTABLE_GIT" -OutFile "$BUILD_DIR\PortableGit.exe"
    $ProgressPreference = 'Continue'            # Subsequent calls do display UI.
    # Extract the contents
    md -Force "$BUILD_DIR\PortableGit"
    Start-Process "$BUILD_DIR\PortableGit.exe" -NoNewWindow -Wait -ArgumentList "-o`"$BUILD_DIR\PortableGit`" -y" 
    
    # # Download depthai
    # git clone https://github.com/luxonis/depthai.git
    #git clone --depth 1 --recurse-submodules --shallow-submodules --branch main https://github.com/luxonis/depthai.git "$BUILD_DIR\depthai"
    # # Install prerequisite dependencies upfront
    # & "$PSScriptRoot\prerequisite.ps1"
    # # Install DepthAI dependencies upfront
    # cd "$PSScriptRoot\depthai"
    # & "$PSScriptRoot\venv\Scripts\python.exe" install_requirements.py
    # cd "$PSScriptRoot"

}
finally {
    # Restore the previous location.
    $prevPwd | Set-Location
}
