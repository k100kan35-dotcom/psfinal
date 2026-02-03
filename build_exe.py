"""
Persson 마찰 모델 - EXE 빌드 스크립트
용량 최소화를 위한 PyInstaller 설정
실행: python build_exe.py
"""
import PyInstaller.__main__
import os
import sys

# 불필요한 모듈 제외 목록 (용량 절감 핵심)
EXCLUDES = [
    # scipy 미사용 서브모듈
    'scipy.stats',
    'scipy.fft',
    'scipy.io',
    'scipy.sparse',
    'scipy.spatial',
    'scipy.cluster',
    'scipy.odr',
    'scipy.constants',
    'scipy.datasets',
    'scipy.misc',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
    # matplotlib 미사용 백엔드/모듈
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
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_pgf',
    'matplotlib.backends.backend_svg',
    'matplotlib.backends.backend_ps',
    # 불필요 패키지
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'setuptools',
    'pip',
    'wheel',
    'pkg_resources',
    'doctest',
    'pydoc',
    'unittest',
    'test',
    'tkinter.test',
    'numpy.testing',
    'scipy.testing',
    'PIL',
    'pillow',
    'email',
    'html',
    'http',
    'xml',
    'xmlrpc',
    'pdb',
    'multiprocessing',
    'concurrent',
    'asyncio',
    'curses',
    'lib2to3',
    'ensurepip',
    'idlelib',
    'distutils',
]

# 필수 hidden imports
HIDDEN_IMPORTS = [
    'numpy',
    'numpy.core',
    'scipy.integrate',
    'scipy.interpolate',
    'scipy.optimize',
    'scipy.signal',
    'scipy.special',
    'scipy.ndimage',
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.figure',
    'tkinter',
    'tkinter.ttk',
    'tkinter.filedialog',
    'tkinter.messagebox',
]

def build():
    args = [
        'main.py',
        '--onefile',
        '--name=PerssonFrictionModel',
        '--clean',
        '--noconfirm',
        # 콘솔 창 숨김 (GUI 앱이므로)
        '--noconsole',
    ]

    # 제외 모듈
    for exc in EXCLUDES:
        args.extend(['--exclude-module', exc])

    # Hidden imports
    for hi in HIDDEN_IMPORTS:
        args.extend(['--hidden-import', hi])

    # persson_model 패키지 포함
    args.extend(['--add-data', 'persson_model:persson_model'])

    # reference_data 포함
    if os.path.isdir('reference_data'):
        args.extend(['--add-data', 'reference_data:reference_data'])

    # 최적화 옵션
    args.extend([
        '--strip',                    # 심볼 제거
        '--log-level', 'WARN',
    ])

    print("=" * 60)
    print("  Persson Friction Model - EXE Build")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Excludes: {len(EXCLUDES)} modules")
    print()

    PyInstaller.__main__.run(args)

    # 결과 확인
    exe_path = os.path.join('dist', 'PerssonFrictionModel')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"\nBuild SUCCESS: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
    else:
        print("\nBuild completed. Check dist/ folder.")


if __name__ == '__main__':
    build()
