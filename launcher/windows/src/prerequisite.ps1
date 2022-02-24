# Use packaged python to create virtualenv
& "$PSScriptRoot\WPy64-3950\python-3.9.5.amd64\python.exe" -m venv venv
& "$PSScriptRoot\venv\Scripts\python.exe" -m pip install -r "$PSScriptRoot\depthai\launcher\requirements.txt"

# # Create a DepthAI.lnk
# .\create_shortcut.ps1
