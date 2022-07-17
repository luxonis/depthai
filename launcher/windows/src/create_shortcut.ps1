# Create a DepthAI.lnk with icon
$Command = "$PSScriptRoot\venv\Scripts\python.exe"
$Arguments = "`"$PSScriptRoot\depthai\launcher\launcher.py`" --repo `"$PSScriptRoot\depthai`" --git `"$PSScriptRoot\PortableGit\bin\git.exe`""
$Ps1File = "$PSScriptRoot\run.ps1"

$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$PSScriptRoot\DepthAI.lnk")
$Shortcut.TargetPath = "powershell"
$Shortcut.Arguments = "-noexit -ExecutionPolicy Bypass -File " + $Ps1File
$Shortcut.IconLocation = "$PSScriptRoot\logo_only_EBl_icon.ico"
$Shortcut.WorkingDirectory = "$PSScriptRoot"
$Shortcut.WindowStyle = 7 # Minimized
$Shortcut.Save()


$StartCommand = "$Command $Arguments"
Set-Content -Path $Ps1File -Value $StartCommand -Encoding ASCII