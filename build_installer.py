"""
NEXEN Rubber Friction Model Program - 인스톨러 빌드 스크립트
==========================================================

사용법 (Windows에서 실행):
    python build_installer.py

사전 설치:
    pip install pyinstaller numpy scipy matplotlib

이 스크립트가 하는 일:
    1) PyInstaller --onedir 모드로 빌드 (--onefile보다 실행 속도 5~10배 빠름)
    2) Inno Setup 스크립트(.iss) 자동 생성
    3) Inno Setup이 설치되어 있으면 자동으로 인스톨러(Setup_*.exe) 컴파일
    4) Inno Setup이 없으면 .iss 파일만 생성 (수동 컴파일 가능)

왜 --onedir인가?
    --onefile: 매 실행마다 임시 폴더에 전체 압축 해제 → 시작 10~30초
    --onedir:  이미 풀려 있는 폴더에서 바로 실행 → 시작 1~3초
    인스톨러로 배포하면 --onedir의 많은 파일 문제도 해결됨
"""

import os
import sys
import shutil
import subprocess
import time
import glob as glob_mod

# ============================================================
# 설정
# ============================================================
APP_NAME = "PerssonFrictionModel"
APP_DISPLAY_NAME = "NEXEN Rubber Friction Model Program"
APP_VERSION = "3.0.0"
APP_PUBLISHER = "NEXEN"
APP_EXE_NAME = f"{APP_NAME}.exe"

# 경로
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(PROJECT_DIR, "dist")
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
OUTPUT_DIR = os.path.join(DIST_DIR, APP_NAME)
ICON_PATH = os.path.join(PROJECT_DIR, "assets", "app_icon.ico")
MAIN_SCRIPT = os.path.join(PROJECT_DIR, "main.py")

# ============================================================
# 프로젝트와 무관한 대형 패키지 제외
# ============================================================
EXCLUDES = [
    # 대형 ML/DL
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
    # 불필요 도구
    'IPython', 'jupyter', 'notebook',
    'pytest',
    # 불필요 matplotlib 백엔드 (TkAgg만 사용)
    'matplotlib.tests',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qt5',
    'matplotlib.backends.backend_qt',
    'matplotlib.backends.backend_qtagg',
    'matplotlib.backends.backend_gtk3',
    'matplotlib.backends.backend_gtk3agg',
    'matplotlib.backends.backend_gtk4',
    'matplotlib.backends.backend_gtk4agg',
    'matplotlib.backends.backend_wx',
    'matplotlib.backends.backend_wxagg',
    'matplotlib.backends.backend_webagg',
    'matplotlib.backends.backend_nbagg',
    'matplotlib.backends.backend_cairo',
    'matplotlib.backends.backend_macosx',
]

# ============================================================
# Hidden imports
# ============================================================
HIDDEN_IMPORTS = [
    # matplotlib 핵심
    'matplotlib', 'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure', 'matplotlib.font_manager',
    'matplotlib.ft2font', 'matplotlib.mathtext',
    'matplotlib._mathtext', 'matplotlib.ticker',
    'matplotlib.colors', 'matplotlib.cm',
    'matplotlib.collections', 'matplotlib.patches',
    'matplotlib.lines', 'matplotlib.image',
    'matplotlib.text', 'matplotlib.legend',
    'matplotlib.scale', 'matplotlib.transforms',
    # numpy / scipy
    'numpy', 'numpy.core', 'numpy.core.multiarray',
    'numpy.core.numeric', 'numpy.fft',
    'scipy', 'scipy.integrate', 'scipy.interpolate',
    'scipy.optimize', 'scipy.signal', 'scipy.special',
    'scipy.signal._savitzky_golay',
    # pandas (선택적이지만 포함)
    'pandas', 'pandas.core',
    # tkinter
    'tkinter', 'tkinter.ttk',
    'tkinter.filedialog', 'tkinter.messagebox',
    'tkinter.simpledialog', 'tkinter.colorchooser',
    # stdlib
    'platform', 'tempfile', 'csv', 're', 'json',
    'importlib', 'importlib.metadata',
    # importlib_resources
    'importlib_resources', 'importlib_resources.trees',
    # pkg_resources / jaraco
    'pkg_resources',
    'jaraco', 'jaraco.text', 'jaraco.functools', 'jaraco.context',
    # persson_model 전체
    'persson_model',
    'persson_model.core',
    'persson_model.core.contact',
    'persson_model.core.friction',
    'persson_model.core.g_calculator',
    'persson_model.core.master_curve',
    'persson_model.core.psd_from_profile',
    'persson_model.core.psd_models',
    'persson_model.core.viscoelastic',
    'persson_model.utils',
    'persson_model.utils.data_loader',
    'persson_model.utils.numerical',
    'persson_model.utils.output',
]


def _print_header():
    print("=" * 64)
    print(f"  {APP_DISPLAY_NAME}")
    print(f"  Installer Build Script v{APP_VERSION}")
    print("=" * 64)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Project : {PROJECT_DIR}")
    print()


def _check_prerequisites():
    """필수 도구 및 파일 확인"""
    errors = []

    # PyInstaller
    try:
        import PyInstaller
        print(f"  [OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        errors.append("PyInstaller가 설치되지 않았습니다: pip install pyinstaller")

    # main.py
    if os.path.isfile(MAIN_SCRIPT):
        print(f"  [OK] {os.path.basename(MAIN_SCRIPT)}")
    else:
        errors.append(f"main.py를 찾을 수 없습니다: {MAIN_SCRIPT}")

    # 아이콘
    if os.path.isfile(ICON_PATH):
        print(f"  [OK] app_icon.ico")
    else:
        print(f"  [!!] app_icon.ico 없음 → 기본 아이콘 사용")
        print(f"       (assets/app_icon.ico 파일을 추가하면 커스텀 아이콘 적용)")

    # 데이터 디렉토리
    for d in ['persson_model', 'assets', 'preset_data', 'reference_data']:
        path = os.path.join(PROJECT_DIR, d)
        if os.path.isdir(path):
            print(f"  [OK] {d}/")
        elif d in ('persson_model',):
            errors.append(f"필수 디렉토리 없음: {d}/")

    print()

    if errors:
        print("  [ERROR] 빌드를 계속할 수 없습니다:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)


def _kill_old_exe():
    """기존 빌드 산출물 정리 (OneDrive 잠금 방지)"""
    exe_path = os.path.join(OUTPUT_DIR, APP_EXE_NAME)
    if not os.path.exists(exe_path):
        return

    if sys.platform == 'win32':
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', APP_EXE_NAME],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(1)
        except FileNotFoundError:
            pass

    for attempt in range(5):
        try:
            os.remove(exe_path)
            print(f"  [CLEAN] Removed old {APP_EXE_NAME}")
            return
        except PermissionError:
            wait = 2 * (attempt + 1)
            print(f"  [WAIT] {APP_EXE_NAME} locked ({attempt+1}/5), waiting {wait}s...")
            time.sleep(wait)
        except FileNotFoundError:
            return


def _build_pyinstaller():
    """PyInstaller --onedir 빌드 실행"""
    import PyInstaller.__main__

    sep = ';' if sys.platform == 'win32' else ':'

    print("-" * 64)
    print("  Step 1: PyInstaller --onedir 빌드")
    print("-" * 64)

    _kill_old_exe()

    args = [
        MAIN_SCRIPT,
        '--onedir',
        f'--name={APP_NAME}',
        '--clean',
        '--noconfirm',
        '--noconsole',
        '--log-level', 'WARN',

        # === matplotlib 데이터 (폰트, matplotlibrc, 스타일 등) 완전 번들 ===
        '--collect-data', 'matplotlib',

        # === jaraco namespace package ===
        '--collect-all', 'jaraco.text',
    ]

    # 아이콘 설정 (있을 때만)
    if os.path.isfile(ICON_PATH):
        args.extend(['--icon', ICON_PATH])

    # DPI-aware manifest (Windows 고해상도 디스플레이에서 UI 스케일링 문제 방지)
    manifest_path = os.path.join(PROJECT_DIR, 'assets', 'app.manifest')
    if os.path.isfile(manifest_path):
        args.extend(['--manifest', manifest_path])

    # Hidden imports
    for mod in HIDDEN_IMPORTS:
        args.extend(['--hidden-import', mod])

    # 제외 모듈
    for exc in EXCLUDES:
        args.extend(['--exclude-module', exc])

    # === 데이터 디렉토리 번들 ===
    data_dirs = {
        'persson_model': 'persson_model',
        'assets': 'assets',
        'reference_data': 'reference_data',
        'preset_data': 'preset_data',
    }
    for src, dst in data_dirs.items():
        src_path = os.path.join(PROJECT_DIR, src)
        if os.path.isdir(src_path):
            args.extend(['--add-data', f'{src_path}{sep}{dst}'])

    # 개별 데이터 파일
    data_files = {
        'strain.py': '.',
        'reference_datasets.json': '.',
    }
    for fname, dst in data_files.items():
        fpath = os.path.join(PROJECT_DIR, fname)
        if os.path.isfile(fpath):
            args.extend(['--add-data', f'{fpath}{sep}{dst}'])

    print()
    print("  Building... (수 분 소요될 수 있습니다)")
    print()

    PyInstaller.__main__.run(args)

    # 빌드 결과 확인
    exe_path = os.path.join(OUTPUT_DIR, APP_EXE_NAME)
    if sys.platform != 'win32':
        exe_path = os.path.join(OUTPUT_DIR, APP_NAME)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print()
        print(f"  [OK] PyInstaller 빌드 성공: {exe_path}")
        print(f"  [OK] EXE 크기: {size_mb:.1f} MB")

        # 전체 폴더 크기
        total = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(OUTPUT_DIR)
            for f in fns
        )
        print(f"  [OK] 배포 폴더 전체: {total / (1024*1024):.1f} MB")
    else:
        print()
        print("  [FAIL] PyInstaller 빌드 실패!")
        print("         로그를 확인하세요.")
        sys.exit(1)

    return exe_path


def _find_inno_setup():
    """Inno Setup 컴파일러(ISCC.exe) 경로 탐색 (Windows 전용)"""
    if sys.platform != 'win32':
        return None

    # 일반적인 설치 경로
    candidates = [
        os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'Inno Setup 6', 'ISCC.exe'),
        os.path.join(os.environ.get('ProgramFiles', ''), 'Inno Setup 6', 'ISCC.exe'),
        os.path.join(os.environ.get('ProgramFiles(x86)', ''), 'Inno Setup 5', 'ISCC.exe'),
        os.path.join(os.environ.get('ProgramFiles', ''), 'Inno Setup 5', 'ISCC.exe'),
    ]

    # PATH에서도 탐색
    iscc_in_path = shutil.which('ISCC') or shutil.which('iscc')
    if iscc_in_path:
        candidates.insert(0, iscc_in_path)

    for path in candidates:
        if path and os.path.isfile(path):
            return path

    return None


def _generate_inno_script():
    """Inno Setup 스크립트(.iss) 생성"""
    print()
    print("-" * 64)
    print("  Step 2: Inno Setup 인스톨러 스크립트 생성")
    print("-" * 64)

    # 아이콘 경로 (Inno Setup용 - 상대경로)
    icon_directive = ""
    if os.path.isfile(ICON_PATH):
        icon_directive = f'SetupIconFile={ICON_PATH}'

    uninstall_icon = ""
    if os.path.isfile(ICON_PATH):
        uninstall_icon = f'UninstallDisplayIcon={{app}}\\{APP_EXE_NAME}'

    iss_content = f"""; NEXEN Rubber Friction Model Program - Inno Setup Script
; 자동 생성됨 - build_installer.py
; Inno Setup 6 이상 필요: https://jrsoftware.org/isinfo.php

[Setup]
AppName={APP_DISPLAY_NAME}
AppVersion={APP_VERSION}
AppPublisher={APP_PUBLISHER}
DefaultDirName={{autopf}}\\{APP_NAME}
DefaultGroupName={APP_DISPLAY_NAME}
OutputDir={DIST_DIR}
OutputBaseFilename=Setup_{APP_NAME}_v{APP_VERSION}
Compression=lzma2/ultra64
SolidCompression=yes
{icon_directive}
{uninstall_icon}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\\Korean.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 생성"; GroupDescription: "추가 작업:"

[Files]
; PyInstaller --onedir 출력 전체를 설치 폴더에 복사
Source: "{OUTPUT_DIR}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\{APP_DISPLAY_NAME}"; Filename: "{{app}}\\{APP_EXE_NAME}"
Name: "{{group}}\\{APP_DISPLAY_NAME} 제거"; Filename: "{{uninstallexe}}"
Name: "{{autodesktop}}\\{APP_DISPLAY_NAME}"; Filename: "{{app}}\\{APP_EXE_NAME}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{APP_EXE_NAME}"; Description: "프로그램 바로 실행"; Flags: nowait postinstall skipifsilent
"""

    iss_path = os.path.join(PROJECT_DIR, f"{APP_NAME}_installer.iss")
    with open(iss_path, 'w', encoding='utf-8-sig') as f:
        f.write(iss_content)

    print(f"  [OK] Inno Setup 스크립트 생성: {iss_path}")
    return iss_path


def _compile_installer(iss_path):
    """Inno Setup으로 인스톨러 컴파일"""
    print()
    print("-" * 64)
    print("  Step 3: 인스톨러 컴파일")
    print("-" * 64)

    iscc = _find_inno_setup()

    if not iscc:
        print()
        print("  [INFO] Inno Setup이 설치되지 않았습니다.")
        print()
        print("  인스톨러를 만들려면:")
        print("    1. https://jrsoftware.org/isdl.php 에서 Inno Setup 6 다운로드 & 설치")
        print("    2. 설치 시 'Install Inno Setup Preprocessor' 체크")
        print("    3. 다음 중 하나를 실행:")
        print(f'       a) 이 스크립트 다시 실행: python build_installer.py')
        print(f'       b) Inno Setup에서 직접 열기: {iss_path}')
        print(f'       c) 커맨드라인: "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe" "{iss_path}"')
        print()
        print("  [참고] Inno Setup 없이도 dist/ 폴더의 EXE는 바로 실행 가능합니다.")
        return None

    print(f"  Inno Setup 발견: {iscc}")
    print(f"  컴파일 중...")
    print()

    result = subprocess.run(
        [iscc, iss_path],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        installer_name = f"Setup_{APP_NAME}_v{APP_VERSION}.exe"
        installer_path = os.path.join(DIST_DIR, installer_name)
        if os.path.exists(installer_path):
            size_mb = os.path.getsize(installer_path) / (1024 * 1024)
            print(f"  [OK] 인스톨러 생성 완료!")
            print(f"  [OK] 파일: {installer_path}")
            print(f"  [OK] 크기: {size_mb:.1f} MB")
            return installer_path
    else:
        print(f"  [FAIL] Inno Setup 컴파일 실패:")
        if result.stdout:
            print(result.stdout[-500:])
        if result.stderr:
            print(result.stderr[-500:])

    return None


def _print_summary(exe_path, installer_path):
    """최종 요약"""
    print()
    print("=" * 64)
    print("  빌드 완료 요약")
    print("=" * 64)
    print()
    print(f"  프로그램 이름 : {APP_DISPLAY_NAME}")
    print(f"  버전         : {APP_VERSION}")
    print()

    if installer_path:
        size_mb = os.path.getsize(installer_path) / (1024 * 1024)
        print(f"  ★ 인스톨러: {installer_path}")
        print(f"    크기: {size_mb:.1f} MB")
        print(f"    → 이 파일을 배포하세요!")
    else:
        print(f"  ★ 실행 파일 폴더: {OUTPUT_DIR}")
        print(f"    → 이 폴더 전체를 압축하여 배포하거나,")
        print(f"      Inno Setup 설치 후 다시 빌드하세요.")

    print()
    print(f"  직접 실행 테스트:")
    print(f"    {exe_path}")
    print()
    print("=" * 64)


def build():
    """메인 빌드 프로세스"""
    os.chdir(PROJECT_DIR)

    _print_header()

    print("  사전 확인:")
    _check_prerequisites()

    exe_path = _build_pyinstaller()

    iss_path = _generate_inno_script()

    installer_path = _compile_installer(iss_path)

    _print_summary(exe_path, installer_path)


if __name__ == '__main__':
    build()
