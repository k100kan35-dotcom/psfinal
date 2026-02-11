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

# 불필요 stdlib/패키지 제외
EXCLUDE_MISC = [
    'IPython', 'jupyter', 'notebook',
    'pytest', 'pip', 'wheel',
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
        # mpl-data/fonts/ (DejaVu Sans 등 내장 .ttf)
        # mpl-data/stylelib/, mpl-data/matplotlibrc
        '--collect-data', 'matplotlib',

        # ===== hidden imports: matplotlib 핵심 =====
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

        # ===== hidden imports: stdlib (동적 import) =====
        '--hidden-import', 'platform',
        '--hidden-import', 'tempfile',
        '--hidden-import', 'csv',
        '--hidden-import', 're',

        # ===== pkg_resources / jaraco 의존성 =====
        # jaraco는 namespace package → collect-all 필수 (collect-submodules 불가)
        '--hidden-import', 'pkg_resources',
        '--collect-all', 'jaraco',
        '--collect-all', 'jaraco.text',
        '--collect-all', 'jaraco.functools',
        '--collect-all', 'jaraco.context',
        '--collect-all', 'importlib_resources',

        # ===== hidden imports: persson_model 패키지 전체 =====
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
    # persson_model 패키지 (핵심 계산 모듈)
    args.extend(['--add-data', f'persson_model{sep}persson_model'])

    # reference_data (검증용 참조 데이터)
    if os.path.isdir('reference_data'):
        args.extend(['--add-data', f'reference_data{sep}reference_data'])

    # preset_data (내장 PSD, aT, mastercurve, strain_sweep, fg_curve)
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

    # 포함 데이터 디렉토리 확인 출력
    for d in ['persson_model', 'reference_data', 'preset_data']:
        if os.path.isdir(d):
            sub = [s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))]
            print(f"  [DATA] {d}/ ({len(sub)} subdirs: {', '.join(sub)})")
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
