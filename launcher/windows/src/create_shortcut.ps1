# Create a DepthAI.lnk with icon
$TargetPath = "$PSScriptRoot\venv\Scripts\python.exe"
$Arguments = "`"$PSScriptRoot\depthai\launcher\launcher.py`" --repo `"$PSScriptRoot\depthai`" --git `"$PSScriptRoot\PortableGit\bin\git.exe`""
$BatFile = "$PSScriptRoot\run.bat"

$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$PSScriptRoot\DepthAI.lnk")
$Shortcut.TargetPath = "cmd"
$Shortcut.Arguments = "/c $BatFile"
$Shortcut.IconLocation = "$PSScriptRoot\logo_only_EBl_icon.ico"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.WindowStyle = 7 # Minimized
$Shortcut.Save()


$StartCommand = "$TargetPath $Arguments"
Set-Content -Path $BatFile -Value $StartCommand -Encoding ASCII