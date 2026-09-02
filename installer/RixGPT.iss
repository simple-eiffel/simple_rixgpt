; Inno Setup script for RixGPT - a standalone char-level LLM (Eiffel EXE + fp16 weights).
; Compile:  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" RixGPT.iss
; Output:   .\RixGPT-Setup.exe

#define AppName "RixGPT"
#define AppVersion "1.0"
#define AppPublisher "Larry Rix"
#define AppExe "RixGPT.exe"

[Setup]
AppId={{FC3DC310-F370-4C5F-AF30-67F3194BE973}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExe}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir=.
OutputBaseFilename=RixGPT-Setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "src\RixGPT.exe";       DestDir: "{app}"; Flags: ignoreversion
Source: "src\model.bin";        DestDir: "{app}"; Flags: ignoreversion
Source: "src\Start RixGPT.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "src\README.txt";       DestDir: "{app}"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\Start RixGPT.bat"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExe}"
Name: "{group}\{#AppName} Read Me";      Filename: "{app}\README.txt"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\Start RixGPT.bat"; WorkingDir: "{app}"; IconFilename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\Start RixGPT.bat"; WorkingDir: "{app}"; Description: "Launch {#AppName} now"; Flags: shellexec postinstall skipifsilent nowait
