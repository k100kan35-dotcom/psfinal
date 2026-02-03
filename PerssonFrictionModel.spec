# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('persson_model', 'persson_model'), ('reference_data', 'reference_data')],
    hiddenimports=['numpy', 'numpy.core', 'scipy.integrate', 'scipy.interpolate', 'scipy.optimize', 'scipy.signal', 'scipy.special', 'matplotlib', 'matplotlib.pyplot', 'matplotlib.backends.backend_tkagg', 'matplotlib.figure', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy.stats', 'scipy.fft', 'scipy.io', 'scipy.sparse', 'scipy.spatial', 'scipy.ndimage', 'scipy.cluster', 'scipy.odr', 'scipy.constants', 'scipy.datasets', 'scipy.misc', 'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack', 'matplotlib.backends.backend_qt5agg', 'matplotlib.backends.backend_qt5', 'matplotlib.backends.backend_qt', 'matplotlib.backends.backend_qtagg', 'matplotlib.backends.backend_gtk3', 'matplotlib.backends.backend_gtk3agg', 'matplotlib.backends.backend_gtk4', 'matplotlib.backends.backend_gtk4agg', 'matplotlib.backends.backend_wx', 'matplotlib.backends.backend_wxagg', 'matplotlib.backends.backend_webagg', 'matplotlib.backends.backend_nbagg', 'matplotlib.backends.backend_cairo', 'matplotlib.backends.backend_macosx', 'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_pgf', 'matplotlib.backends.backend_svg', 'matplotlib.backends.backend_ps', 'IPython', 'jupyter', 'notebook', 'pytest', 'setuptools', 'pip', 'wheel', 'pkg_resources', 'doctest', 'pydoc', 'unittest', 'test', 'tkinter.test', 'numpy.testing', 'scipy.testing', 'PIL', 'pillow', 'email', 'html', 'http', 'xml', 'xmlrpc', 'pdb', 'multiprocessing', 'concurrent', 'asyncio', 'curses', 'lib2to3', 'ensurepip', 'idlelib', 'distutils'],
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
