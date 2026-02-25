; Persson Friction Model - Inno Setup Installer Script
; ====================================================
; 사용법:
;   1. PyInstaller로 먼저 빌드: python build_exe.py --onedir
;   2. Inno Setup으로 컴파일: iscc installer.iss
;
; Inno Setup 다운로드: https://jrsoftware.org/issetup.php

#define MyAppName "Persson Friction Model"
#define MyAppVersion "1.1.0"
#define MyAppPublisher "Persson Modelling Team"
#define MyAppExeName "PerssonFrictionModel.exe"
#define MyAppDescription "Persson Contact Mechanics and Friction Theory"

[Setup]
; 고유 앱 ID (GUID) - 이 값은 변경하지 마세요
AppId={{B3F7A2D1-8E4C-4F9A-B6D2-1A3E5C7F9B0D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppComments={#MyAppDescription}
DefaultDirName={autopf}\PerssonFrictionModel
DefaultGroupName={#MyAppName}
; 설치 마법사 설정
AllowNoIcons=yes
; 출력 설치 파일 설정
OutputDir=installer_output
OutputBaseFilename=PerssonFrictionModel_v{#MyAppVersion}_Setup
; 압축 설정 (LZMA2 최대 압축)
Compression=lzma2/ultra64
SolidCompression=yes
; 최소 Windows 버전 (Windows 7 SP1 이상)
MinVersion=6.1sp1
; 관리자 권한 불필요 (사용자 폴더 설치 가능)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; 64비트 모드
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; UI 설정
WizardStyle=modern
; 설치 중 앱 닫기 지원
CloseApplications=yes
RestartApplications=no
; 제거 시 설정
UninstallDisplayName={#MyAppName}

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; PyInstaller onedir 출력물 전체 포함
Source: "dist\PerssonFrictionModel\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; 프리셋 데이터 (별도 복사 - onedir에 미포함 시)
Source: "preset_data\*"; DestDir: "{app}\preset_data"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: DirExists(ExpandConstant('{src}\preset_data'))
; 참조 데이터
Source: "reference_data\*"; DestDir: "{app}\reference_data"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: DirExists(ExpandConstant('{src}\reference_data'))

[Icons]
; 시작 메뉴 바로가기
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyAppDescription}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
; 바탕화면 바로가기
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyAppDescription}"

[Run]
; 설치 완료 후 실행 옵션
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(#MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// 설치 전 이전 버전 확인 및 제거
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
