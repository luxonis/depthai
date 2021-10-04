# Downloads and installs Inno Setup 6.2.0
wget "https://files.jrsoftware.org/is/6/innosetup-6.2.0.exe" -o "$PSScriptRoot\is.exe"
Start-Process "$PSScriptRoot\is.exe" -NoNewWindow -Wait -ArgumentList "/SP- /VERYSILENT /ALLUSERS /SUPPRESSMSGBOXES"
