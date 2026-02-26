"""
Persson 마찰 모델 - Windows EXE 빌드 스크립트
Windows에서 실행: python build_exe.py
설치 파일용 빌드: python build_exe.py --onedir
사전 설치: pip install pyinstaller numpy scipy matplotlib pandas
"""
import PyInstaller.__main__
import os
import sys
import shutil
import subprocess
import time

APP_NAME = "NexenRubberFriction"
APP_VERSION = "3.0"

# 프로젝트와 100% 무관한 대형 ML/DL 패키지만 제외
EXCLUDES = [
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
]

# matplotlib 불필요 백엔드/테스트 제외 (TkAgg만 사용)
EXCLUDE_MPL = [
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

# 불필요 외부 패키지만 제외
# 주의: stdlib 모듈(unittest, pydoc, doctest 등)은 절대 제외 금지
#   → numpy.testing → unittest, scipy._lib._docscrape → pydoc 등
#   예측 불가능한 내부 의존성이 존재함
# 주의: wheel, setuptools, pip 도 제외 금지 (PyInstaller hook 충돌)
EXCLUDE_MISC = [
    'IPython', 'jupyter', 'notebook',
    'pytest',
]

def _kill_old_exe(exe_path):
    """빌드 전 기존 EXE를 정리 (OneDrive 동기화 잠금 방지)"""
    if not os.path.exists(exe_path):
        return

    exe_name = os.path.basename(exe_path)

    # 1) 실행 중인 EXE 프로세스 강제 종료 (Windows)
    if sys.platform == 'win32':
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', exe_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1)
        except FileNotFoundError:
            pass

    # 2) 기존 EXE 삭제 시도 (최대 5회, OneDrive 해제 대기)
    for attempt in range(5):
        try:
            os.remove(exe_path)
            print(f"  [CLEAN] Removed old {exe_name}")
            return
        except PermissionError:
            wait = 2 * (attempt + 1)
            print(f"  [WAIT] {exe_name} locked (attempt {attempt+1}/5), "
                  f"waiting {wait}s for OneDrive to release...")
            time.sleep(wait)
        except FileNotFoundError:
            return

    # 3) 그래도 안 되면 dist 폴더 자체를 교체
    print(f"  [WARN] Cannot delete {exe_name}. Renaming dist/ and creating fresh one.")
    dist_dir = os.path.dirname(exe_path)
    backup = dist_dir + f'_old_{int(time.time())}'
    try:
        os.rename(dist_dir, backup)
    except OSError:
        pass


def _create_dpi_manifest():
    """Windows DPI-aware manifest 파일을 생성합니다.
    이 매니페스트가 있어야 ctypes.windll.user32.GetDpiForSystem()이
    실제 DPI를 반환하고, UI가 올바르게 스케일링됩니다."""
    manifest_content = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0"
          xmlns:asmv3="urn:schemas-microsoft-com:asm.v3">
  <asmv3:application>
    <asmv3:windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">permonitorv2,permonitor,system</dpiAwareness>
    </asmv3:windowsSettings>
  </asmv3:application>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <!-- Windows 10 / 11 -->
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
      <!-- Windows 8.1 -->
      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"/>
      <!-- Windows 8 -->
      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"/>
      <!-- Windows 7 -->
      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"/>
    </application>
  </compatibility>
</assembly>"""

    manifest_path = os.path.join('assets', 'app.manifest')
    os.makedirs('assets', exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(manifest_content)
    print(f"  [DPI] Created DPI-aware manifest: {manifest_path}")
    return manifest_path


def build():
    # --onedir 옵션 확인 (설치 파일 빌드용)
    use_onedir = '--onedir' in sys.argv

    sep = ';' if sys.platform == 'win32' else ':'

    # ===== 빌드 전 기존 EXE 정리 (OneDrive 잠금 방지) =====
    exe_name = f'{APP_NAME}.exe' if sys.platform == 'win32' else APP_NAME
    if use_onedir:
        exe_path = os.path.join('dist', APP_NAME, exe_name)
    else:
        exe_path = os.path.join('dist', exe_name)
    _kill_old_exe(exe_path)

    pack_mode = '--onedir' if use_onedir else '--onefile'

    # DPI-aware manifest 생성 (Windows)
    manifest_path = _create_dpi_manifest()

    args = [
        'main.py',
        pack_mode,
        f'--name={APP_NAME}',
        '--clean',
        '--noconfirm',
        '--noconsole',
        '--icon=assets/app_icon.ico',
        f'--manifest={manifest_path}',
        '--log-level', 'WARN',

        # ===== matplotlib 폰트/데이터 번들 =====
        '--collect-data', 'matplotlib',

        # ===== hidden imports: matplotlib =====
        '--hidden-import', 'matplotlib',
        '--hidden-import', 'matplotlib.pyplot',
        '--hidden-import', 'matplotlib.backends.backend_tkagg',
        '--hidden-import', 'matplotlib.figure',
        '--hidden-import', 'matplotlib.font_manager',
        '--hidden-import', 'matplotlib.ft2font',
        '--hidden-import', 'matplotlib.mathtext',
        '--hidden-import', 'matplotlib._mathtext',
        '--hidden-import', 'matplotlib.ticker',
        '--hidden-import', 'matplotlib.colors',
        '--hidden-import', 'matplotlib.cm',
        '--hidden-import', 'matplotlib.collections',

        # ===== hidden imports: numpy/scipy =====
        '--hidden-import', 'numpy',
        '--hidden-import', 'numpy.core',
        '--hidden-import', 'scipy.integrate',
        '--hidden-import', 'scipy.interpolate',
        '--hidden-import', 'scipy.optimize',
        '--hidden-import', 'scipy.signal',
        '--hidden-import', 'scipy.special',

        # ===== hidden imports: pandas =====
        '--hidden-import', 'pandas',
        '--hidden-import', 'pandas.core',

        # ===== hidden imports: tkinter =====
        '--hidden-import', 'tkinter',
        '--hidden-import', 'tkinter.ttk',
        '--hidden-import', 'tkinter.filedialog',
        '--hidden-import', 'tkinter.messagebox',
        '--hidden-import', 'tkinter.simpledialog',
        '--hidden-import', 'tkinter.font',

        # ===== hidden imports: stdlib (DPI 스케일링 등) =====
        '--hidden-import', 'platform',
        '--hidden-import', 'tempfile',
        '--hidden-import', 'csv',
        '--hidden-import', 're',
        '--hidden-import', 'ctypes',

        # ===== hidden imports: importlib_resources =====
        '--hidden-import', 'importlib_resources',

        # ===== pkg_resources / jaraco 의존성 =====
        # jaraco는 namespace package → hidden-import만 사용
        # (collect-all/collect-submodules는 namespace pkg에서 크래시)
        '--hidden-import', 'pkg_resources',
        '--hidden-import', 'jaraco',
        '--hidden-import', 'jaraco.text',
        '--hidden-import', 'jaraco.functools',
        '--hidden-import', 'jaraco.context',
        '--collect-all', 'jaraco.text',

        # ===== hidden imports: persson_model =====
        '--hidden-import', 'persson_model',
        '--hidden-import', 'persson_model.core',
        '--hidden-import', 'persson_model.core.contact',
        '--hidden-import', 'persson_model.core.friction',
        '--hidden-import', 'persson_model.core.g_calculator',
        '--hidden-import', 'persson_model.core.master_curve',
        '--hidden-import', 'persson_model.core.psd_from_profile',
        '--hidden-import', 'persson_model.core.psd_models',
        '--hidden-import', 'persson_model.core.viscoelastic',
        '--hidden-import', 'persson_model.utils',
        '--hidden-import', 'persson_model.utils.data_loader',
        '--hidden-import', 'persson_model.utils.numerical',
        '--hidden-import', 'persson_model.utils.output',
    ]

    # 제외 모듈 추가
    for exc in EXCLUDES + EXCLUDE_MPL + EXCLUDE_MISC:
        args.extend(['--exclude-module', exc])

    # ===== 데이터 디렉토리 포함 =====
    if os.path.isdir('assets'):
        args.extend(['--add-data', f'assets{sep}assets'])

    args.extend(['--add-data', f'persson_model{sep}persson_model'])

    if os.path.isdir('reference_data'):
        args.extend(['--add-data', f'reference_data{sep}reference_data'])

    if os.path.isdir('preset_data'):
        args.extend(['--add-data', f'preset_data{sep}preset_data'])

    if os.path.isfile('strain.py'):
        args.extend(['--add-data', f'strain.py{sep}.'])

    print("=" * 60)
    print(f"  NEXEN Rubber Friction Modelling Program v{APP_VERSION} - EXE Build")
    print(f"  Mode: {'onedir (for installer)' if use_onedir else 'onefile (standalone)'}")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    for d in ['persson_model', 'reference_data', 'preset_data']:
        if os.path.isdir(d):
            sub = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            print(f"  [DATA] {d}/ ({len(sub)} subdirs: {', '.join(sub)})")
    print()

    PyInstaller.__main__.run(args)

    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\nBuild SUCCESS: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
    else:
        print("\nBuild completed. Check dist/ folder.")


if __name__ == '__main__':
    build()
