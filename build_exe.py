"""
Persson 마찰 모델 - Windows EXE 빌드 스크립트
Windows에서 실행: python build_exe.py
사전 설치: pip install pyinstaller numpy scipy matplotlib
"""
import PyInstaller.__main__
import os
import sys

# 이 프로젝트와 무관한 대형 패키지만 제외 (numpy/scipy/matplotlib/tkinter는 전부 포함)
EXCLUDES = [
    # 머신러닝/딥러닝 (프로젝트 미사용, 각각 수백MB~수GB)
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
    'sklearn', 'scikit-learn',
    # 데이터/웹 (프로젝트 미사용)
    'pandas', 'openpyxl', 'xlrd', 'xlsxwriter',
    'flask', 'django', 'fastapi', 'uvicorn',
    'requests', 'urllib3', 'httpx',
    'sqlalchemy', 'sqlite3',
    'boto3', 'botocore',
    # Jupyter/개발도구 (프로젝트 미사용)
    'IPython', 'jupyter', 'notebook', 'nbconvert', 'nbformat',
    'debugpy', 'black', 'flake8', 'mypy', 'pylint',
    'pytest', 'coverage',
    'sphinx',
    # 이미지/미디어 (프로젝트 미사용)
    'PIL', 'pillow', 'cv2', 'opencv',
    'imageio', 'skimage',
    # 기타 대형 패키지
    'sympy',
    'dask',
    'h5py', 'tables',
    'zmq', 'tornado',
    'cryptography',
    'paramiko',
    'docker',
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
    ]

    # 무관한 대형 패키지만 제외
    for exc in EXCLUDES:
        args.extend(['--exclude-module', exc])

    # persson_model 패키지 포함
    args.extend(['--add-data', f'persson_model{sep}persson_model'])

    # reference_data 포함
    if os.path.isdir('reference_data'):
        args.extend(['--add-data', f'reference_data{sep}reference_data'])

    print("=" * 60)
    print("  Persson Friction Model - EXE Build")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Excludes: {len(EXCLUDES)} unrelated packages")
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
