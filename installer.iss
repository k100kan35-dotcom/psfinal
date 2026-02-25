; Persson Friction Model - Inno Setup Installer Script
; ====================================================
; Usage:
;   1. Build with PyInstaller first: python build_exe.py --onedir
;   2. Compile with Inno Setup: iscc installer.iss
;
; Download Inno Setup: https://jrsoftware.org/issetup.php

#define MyAppName "Persson Friction Model"
#define MyAppVersion "1.1.0"
#define MyAppPublisher "Persson Modelling Team"
#define MyAppExeName "PerssonFrictionModel.exe"
#define MyAppDescription "Persson Contact Mechanics and Friction Theory"

[Setup]
; Unique App ID (GUID) - Do not change this value
AppId={{B3F7A2D1-8E4C-4F9A-B6D2-1A3E5C7F9B0D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppComments={#MyAppDescription}
DefaultDirName={autopf}\PerssonFrictionModel
DefaultGroupName={#MyAppName}
; Installer wizard settings
AllowNoIcons=yes
; Output installer file settings
OutputDir=installer_output
OutputBaseFilename=PerssonFrictionModel_v{#MyAppVersion}_Setup
; Compression (LZMA2 max)
Compression=lzma2/ultra64
SolidCompression=yes
; Minimum Windows version (Windows 7 SP1+)
MinVersion=6.1sp1
; No admin privileges required (can install to user folder)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; 64-bit mode
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; UI settings
WizardStyle=modern
; Close running app during install
CloseApplications=yes
RestartApplications=no
; Uninstall display name
UninstallDisplayName={#MyAppName}

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Include entire PyInstaller onedir output
Source: "dist\PerssonFrictionModel\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu shortcuts
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyAppDescription}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; Desktop shortcut
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyAppDescription}"

[Run]
; Option to launch after install
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Check for previous version and offer to uninstall
function InitializeSetup(): Boolean;
var
  UninstallKey: String;
  UninstallString: String;
  ResultCode: Integer;
begin
  Result := True;

  UninstallKey := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1';

  if RegQueryStringValue(HKLM, UninstallKey, 'UninstallString', UninstallString) or
     RegQueryStringValue(HKCU, UninstallKey, 'UninstallString', UninstallString) then
  begin
    if MsgBox('{#MyAppName}' + #13#10 +
              'A previous version is already installed.' + #13#10 +
              'Do you want to uninstall it before continuing?' + #13#10#13#10 +
              '(Recommended: Yes)',
              mbConfirmation, MB_YESNO) = IDYES then
    begin
      Exec(RemoveQuotes(UninstallString), '/SILENT', '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;
