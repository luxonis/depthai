# Downloads and installs Inno Setup 6.2.0
$ProgressPreference = 'SilentlyContinue'    # Subsequent calls do not display UI.
Invoke-WebRequest "https://files.jrsoftware.org/is/6/innosetup-6.2.0.exe" -OutFile "$PSScriptRoot\is.exe"
$ProgressPreference = 'Continue'            # Subsequent calls do display UI.
Start-Process "$PSScriptRoot\is.exe" -NoNewWindow -Wait -ArgumentList "/SP- /VERYSILENT /ALLUSERS /SUPPRESSMSGBOXES"
