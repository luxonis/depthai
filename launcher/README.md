# DepthAI Launcher

DepthAI Launcher is a small utility that provides installation and updates for DepthAI Demo Application

## Windows

DepthAI Launcher includes installation setup for Windows (64bit only).
Installation carries an embedded Python distribution WinPython, DepthAI Launcher and `depthai` repository.

### Troubleshooting

See the logging file by navigating to `%temp%` directory and searching for `Setup Log` files.
(Example log path: `C:\Users\[username]\AppData\Local\Temp\Setup Log 2022-01-28 #001.txt`)

Or run the setup by manually providing the log file location:
```
.\depthai_setup.exe /LOG=C:\Users\[username]\Desktop\depthai_setup.log
```

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
