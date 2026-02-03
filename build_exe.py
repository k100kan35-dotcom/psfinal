"""
Persson 마찰 모델 - Windows EXE 빌드 스크립트
Windows에서 실행: python build_exe.py
사전 설치: pip install pyinstaller numpy scipy matplotlib pandas
"""
import PyInstaller.__main__
import os
import sys

# 프로젝트와 100% 무관한 대형 ML/DL 패키지만 제외
# (numpy, scipy, matplotlib, pandas, tkinter, PIL 등 나머지는 전부 포함)
EXCLUDES = [
    'torch', 'torchvision', 'torchaudio',
    'tensorflow', 'keras',
    'numba', 'llvmlite',
    'tensorboard', 'tensorboardX',
    'onnx', 'onnxruntime',
    'xgboost', 'lightgbm', 'catboost',
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
        # matplotlib 폰트/스타일 데이터 번들 (한글 깨짐 방지)
        '--collect-data', 'matplotlib',
        '--hidden-import', 'matplotlib.backends.backend_tkagg',
    ]

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
