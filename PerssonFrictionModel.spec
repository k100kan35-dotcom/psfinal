# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

# matplotlib 데이터 (폰트, 스타일, matplotlibrc)
mpl_datas = collect_data_files('matplotlib')

# matplotlib 서브모듈 (font_manager, ft2font, mathtext 등)
mpl_hiddenimports = collect_submodules('matplotlib')

# 프로젝트 데이터 디렉토리
project_datas = [
    ('persson_model', 'persson_model'),
    ('reference_data', 'reference_data'),
    ('assets', 'assets'),
]
if os.path.isdir('preset_data'):
    project_datas.append(('preset_data', 'preset_data'))
if os.path.isfile('strain.py'):
    project_datas.append(('strain.py', '.'))
if os.path.isfile('reference_datasets.json'):
    project_datas.append(('reference_datasets.json', '.'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=project_datas + mpl_datas,
    hiddenimports=mpl_hiddenimports + [
        # numpy/scipy
        'numpy', 'numpy.core', 'numpy.core.multiarray',
        'numpy.core.numeric', 'numpy.fft',
        'scipy', 'scipy.integrate', 'scipy.interpolate', 'scipy.optimize',
        'scipy.signal', 'scipy.special',
        'scipy.signal._savitzky_golay',
        # pandas (DMA/PSD 파일 로딩)
        'pandas', 'pandas.core',
        # tkinter
        'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
        # 불필요 외부 패키지
        'IPython', 'jupyter', 'notebook',
        'pytest',
        # 주의: setuptools, pip, wheel, unittest, pydoc, doctest 등 stdlib은 제외 금지
        #   → numpy.testing → unittest, scipy._lib._docscrape → pydoc 등
        #   예측 불가능한 내부 의존성이 존재함
        # 대형 ML/DL
        'torch', 'torchvision', 'torchaudio',
        'tensorflow', 'keras',
        'numba', 'llvmlite',
        'tensorboard', 'tensorboardX',
        'onnx', 'onnxruntime',
        'xgboost', 'lightgbm', 'catboost',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PerssonFrictionModel',
    icon='assets/app_icon.ico' if os.path.isfile('assets/app_icon.ico') else None,
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
