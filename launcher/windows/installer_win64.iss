#define MyAppName "DepthAI"
; Stores version "MyAppVersion" define
#include "version.txt"
#define MyAppPublisher "Luxonis"
#define MyAppURL "https://www.luxonis.com/"
#define MyAppExeName "DepthAI.lnk"
#define MyAppIconName "logo_only_EBl_icon.ico"

; Helper to install all files including hidden files
; Skips folder called "launcher"
#pragma parseroption -p-

; If the file is found by calling FindFirst without faHidden, it's not hidden
#define FileParams(FileName) \
    Local[0] = FindFirst(FileName, 0), \
    (!Local[0] ? "; Attribs: hidden" : "")

#define FileEntry(Source, DestDir) \
    "Source: \"" + Source + "\"; DestDir: \"" + DestDir + "\"" + \
    FileParams(Source) + "\n"

#define ProcessFile(Source, DestDir, FindResult, FindHandle) \
    FindResult \
        ? \
            Local[0] = FindGetFileName(FindHandle), \
            Local[1] = Source + "\\" + Local[0], \
            (Local[0] != "." && Local[0] != ".." && Local[0] != "windows" \
                ? (DirExists(Local[1]) \
                      ? ProcessFolder(Local[1], DestDir + "\\" + Local[0]) \
                      : FileEntry(Local[1], DestDir)) \
                : "") + \
            ProcessFile(Source, DestDir, FindNext(FindHandle), FindHandle) \
        : \
            ""

#define ProcessFolder(Source, DestDir) \
    Local[0] = FindFirst(Source + "\\*", faAnyFile), \
    ProcessFile(Source, DestDir, Local[0], Local[0])

#pragma parseroption -p+


[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{6C2FE7A9-8A5C-4F45-B151-DDDC2A523590}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; Remove the following line to run in administrative install mode (install for all users.)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
OutputBaseFilename=DepthAI_setup
SetupIconFile=../{#MyAppIconName}
UninstallDisplayIcon=../{#MyAppIconName}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
; Output build location
OutputDir=build\Output
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; 

[Files]
; Source: "build\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

; Install 'depthai' repo (/w hidden files)
; Source: "..\..\*"; DestDir: "{app}\depthai"; Flags: ignoreversion recursesubdirs createallsubdirs
#emit ProcessFolder("..\..", "{app}\depthai")

; Install embedded Python
Source: "build\WPy64-3950\*"; DestDir: "{app}\WPy64-3950"; Flags: ignoreversion recursesubdirs createallsubdirs
; Install Portable Git
Source: "build\PortableGit\*"; DestDir: "{app}\PortableGit"; Flags: ignoreversion recursesubdirs createallsubdirs
; Installs venv as well (TBD)
; Source: "build\venv\*"; DestDir: "{app}\venv"; Flags: ignoreversion recursesubdirs createallsubdirs

; Install Windows specific scripts
Source: "src\create_shortcut.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\prerequisite.ps1"; DestDir: "{app}"; Flags: ignoreversion

; ; Install launcher sources
Source: "..\{#MyAppIconName}"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\launcher.py"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\splash2.png"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\demo_card.png"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\viewer_card.png"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\splash_screen.py"; DestDir: "{app}"; Flags: ignoreversion
; Source: "..\choose_app_dialog.py"; DestDir: "{app}"; Flags: ignoreversion
; ; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "cmd.exe"; Parameters: "/C"; StatusMsg: "Installing requirements..."; BeforeInstall: InstallPrerequisite
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: shellexec postinstall skipifsilent

; Creates DepthAI shortcut in installation directory, before installing it as a Desktop Icon
; And uninstall previously installed launcher
[Code]

function GetUninstallString(): String;
var
  sUnInstPath: String;
  sUnInstallString: String;
begin
  sUnInstPath := ExpandConstant('Software\Microsoft\Windows\CurrentVersion\Uninstall\{#emit SetupSetting("AppId")}_is1');
  sUnInstallString := '';
  if not RegQueryStringValue(HKLM, sUnInstPath, 'UninstallString', sUnInstallString) then
    RegQueryStringValue(HKCU, sUnInstPath, 'UninstallString', sUnInstallString);
  Result := sUnInstallString;
end;

function IsUpgrade(): Boolean;
begin
  Result := (GetUninstallString() <> '');
end;

function UnInstallOldVersion(): Integer;
var
  sUnInstallString: String;
  iResultCode: Integer;
begin
; { Return Values: }
; { 1 - uninstall string is empty }
; { 2 - error executing the UnInstallString }
; { 3 - successfully executed the UnInstallString }

  { default return value }
  Result := 0;

  ; { get the uninstall string of the old app }
  sUnInstallString := GetUninstallString();
  if sUnInstallString <> '' then begin
    sUnInstallString := RemoveQuotes(sUnInstallString);
    if Exec(sUnInstallString, '/SILENT /NORESTART /SUPPRESSMSGBOXES','', SW_HIDE, ewWaitUntilTerminated, iResultCode) then
      Result := 3
    else
      Result := 2;
  end else
    Result := 1;
end;

{* Handles uninstallation if previous DepthAI version is installed and shortcut creation *}
procedure CurStepChanged(CurStep: TSetupStep);
var
    TmpFileName: string;
    ExecStdout: AnsiString;
    UtfStdout: string;
    ExecParameter: string;
    ResultCode: Integer;
begin
  if (CurStep=ssInstall) then
  begin
    if (IsUpgrade()) then
    begin
      UnInstallOldVersion();
    end;
  
    Log('Creating main shortcut');
    ExtractTemporaryFile('create_shortcut.ps1');
    ForceDirectories(ExpandConstant('{app}'));
    FileCopy(ExpandConstant('{tmp}\create_shortcut.ps1'), ExpandConstant('{app}\create_shortcut.ps1'), False);
    TmpFileName := ExpandConstant('{tmp}') + '\exec_stdout_stderr_tmp.txt';
    ExecParameter := '/C powershell.exe -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\create_shortcut.ps1') + '" > "' + TmpFileName + '" 2>&1';
    Log('Calling cmd.exe ' + ExecParameter)
    Exec('cmd.exe', ExecParameter, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
    if LoadStringFromFile(TmpFileName, ExecStdout) then begin
      UtfStdout := ExecStdout
      Log(UtfStdout);
    end;
    DeleteFile(TmpFileName);

  end;
end;

procedure InstallPrerequisite;
var
    TmpFileName: string;
    ExecStdout: AnsiString;
    UtfStdout: string;
    ExecParameter: string;
    ResultCode: Integer;
begin
  Log('Installing prerequisites');
  TmpFileName := ExpandConstant('{tmp}') + '\exec_stdout_stderr_tmp.txt';
  ExecParameter := '/C powershell.exe -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\prerequisite.ps1') + '" > "' + TmpFileName + '" 2>&1"';
  Log('Calling cmd.exe ' + ExecParameter)
  Exec('cmd.exe', ExecParameter, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if LoadStringFromFile(TmpFileName, ExecStdout) then begin
    UtfStdout := ExecStdout
    Log(UtfStdout);
  end;
  DeleteFile(TmpFileName);

end;


[UninstallDelete]
Type: filesandordirs; Name: "{app}"
