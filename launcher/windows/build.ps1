# Constants
# Build directory
$BUILD_DIR = "$PSScriptRoot\build"
# WinPython embedded Python3.9
$EMBEDDED_PYTHON="https://github.com/winpython/winpython/releases/download/4.3.20210620/Winpython64-3.9.5.0dot.exe"

# Save the current location and switch to this script's directory.
# Note: This shouldn't fail; if it did, it would indicate a
#       serious system-wide problem.
$prevPwd = $PWD; Set-Location -ErrorAction Stop -LiteralPath $PSScriptRoot

try {

    # Download dependencies
    .\download_dependencies.ps1

    # Build the installer
    ISCC.exe .\installer_win64.iss

}
finally {
    # Restore the previous location.
    $prevPwd | Set-Location
}
