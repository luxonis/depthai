# Create a DepthAI.lnk with icon
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$PSScriptRoot\DepthAI.lnk")
$Shortcut.TargetPath = "$PSScriptRoot\venv\Scripts\python.exe"
$Shortcut.Arguments = "`"$PSScriptRoot\depthai\launcher\launcher.py`" --repo `"$PSScriptRoot\depthai`" --git `"$PSScriptRoot\PortableGit\bin\git.exe`""
$Shortcut.IconLocation = "$PSScriptRoot\logo_only_EBl_icon.ico"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.WindowStyle = 7 # Minimized
$Shortcut.Save()