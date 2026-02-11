"""
Persson 마찰 모델 - Windows EXE 빌드 스크립트
Windows에서 실행: python build_exe.py
사전 설치: pip install pyinstaller numpy scipy matplotlib pandas
"""
import PyInstaller.__main__
import os
import sys

# 프로젝트와 100% 무관한 대형 ML/DL 패키지만 제외
EXCLUDES = [
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
]

# matplotlib 불필요 백엔드 제외 (TkAgg만 사용)
EXCLUDE_MPL_BACKENDS = [
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

# 불필요 stdlib/패키지 제외
EXCLUDE_MISC = [
    'IPython', 'jupyter', 'notebook',
    'pytest', 'setuptools', 'pip', 'wheel',
    'pdb', 'doctest', 'pydoc', 'unittest', 'test',
    'lib2to3', 'ensurepip', 'idlelib', 'distutils',
    'curses',
]

def build():
    sep = ';' if sys.platform == 'win32' else ':'

    args = [
        'main.py',
        '--onefile',
        '--name=PerssonFrictionModel',
        '--clean',
        '--noconfirm',
        '--noconsole',
        '--log-level', 'WARN',

        # ===== matplotlib 폰트/데이터 완전 번들 =====
        # collect-data: matplotlib의 모든 데이터 파일 포함
        #   - mpl-data/fonts/ (DejaVu Sans 등 내장 폰트 .ttf)
        #   - mpl-data/stylelib/ (스타일 파일)
        #   - mpl-data/matplotlibrc (기본 설정)
        '--collect-data', 'matplotlib',
        # collect-submodules: 폰트 렌더링 관련 서브모듈 전체 포함
        #   - matplotlib.font_manager (폰트 탐색/등록)
        #   - matplotlib.ft2font (FreeType2 바인딩)
        #   - matplotlib.mathtext (수식 렌더링)
        #   - matplotlib._mathtext (수식 파서)
        '--collect-submodules', 'matplotlib',

        # ===== 필수 hidden imports =====
        # matplotlib 핵심
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

        # numpy/scipy
        '--hidden-import', 'numpy',
        '--hidden-import', 'numpy.core',
        '--hidden-import', 'scipy.integrate',
        '--hidden-import', 'scipy.interpolate',
        '--hidden-import', 'scipy.optimize',
        '--hidden-import', 'scipy.signal',
        '--hidden-import', 'scipy.special',

        # pandas (main.py에서 DMA/PSD 파일 로딩에 사용)
        '--hidden-import', 'pandas',
        '--hidden-import', 'pandas.core',

        # tkinter GUI
        '--hidden-import', 'tkinter',
        '--hidden-import', 'tkinter.ttk',
        '--hidden-import', 'tkinter.filedialog',
        '--hidden-import', 'tkinter.messagebox',

        # 표준 라이브러리 (한글 폰트 탐색에 필요)
        '--hidden-import', 'platform',
    ]

    # 제외 모듈 추가
    for exc in EXCLUDES + EXCLUDE_MPL_BACKENDS + EXCLUDE_MISC:
        args.extend(['--exclude-module', exc])

    # ===== 데이터 디렉토리 포함 =====
    # persson_model 패키지 (핵심 계산 모듈)
    args.extend(['--add-data', f'persson_model{sep}persson_model'])

    # reference_data (검증용 참조 데이터)
    if os.path.isdir('reference_data'):
        args.extend(['--add-data', f'reference_data{sep}reference_data'])

    # preset_data (내장 PSD, aT, mastercurve 데이터)
    if os.path.isdir('preset_data'):
        args.extend(['--add-data', f'preset_data{sep}preset_data'])

    # strain.py (Strain sweep GUI - 별도 실행 가능)
    if os.path.isfile('strain.py'):
        args.extend(['--add-data', f'strain.py{sep}.'])

    print("=" * 60)
    print("  Persson Friction Model - EXE Build")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()

    PyInstaller.__main__.run(args)

    # 결과 확인
    exe_name = 'PerssonFrictionModel.exe' if sys.platform == 'win32' else 'PerssonFrictionModel'
    exe_path = os.path.join('dist', exe_name)
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\nBuild SUCCESS: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
    else:
        print("\nBuild completed. Check dist/ folder.")


if __name__ == '__main__':
    build()
