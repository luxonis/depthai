# DepthAI Launcher

DepthAI Launcher is a small utility that provides installation and updates for DepthAI Demo Application

## Windows

DepthAI Launcher includes installation setup for Windows (64bit only).
Installation carries an embedded Python distribution WinPython, DepthAI Launcher and `depthai` repository.

### Installer

In the following steps, building the Windows installer from source is presented.

#### Dependencies

The following dependencies are required
 - Windows Host machine (x64)
 - Inno Setup 6.2

#### Building

To build Windows installer, Inno Setup installation directory must be present in `PATH` environmental variable (`ISCC.exe` must be present in the directory).

Execute the `launcher/windows/build.ps1` script to create the Windows installer.
The built installer `DepthAI_setup.exe` can be found in `build/Output/`.
