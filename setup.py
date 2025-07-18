from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import glob

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files*/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files*/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

# Make sure we can find the necessary compiler binaries on Windows
if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
    compiler_bindir = _find_compiler_bindir()
    if compiler_bindir is None:
        raise RuntimeError('Could not find MSVC/GCC/CLANG installation on this computer.')
    os.environ['PATH'] += ';' + compiler_bindir

# Default architectures for broad RTX support including future cards like RTX5090
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.0;7.5;8.0;8.6')

ext_modules = [
    CUDAExtension(
        'bias_act_plugin',
        sources=[
            'torch_utils/ops/bias_act.cpp',
            'torch_utils/ops/bias_act.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '--allow-unsupported-compiler'],
        },
    ),
    CUDAExtension(
        'filtered_lrelu_plugin',
        sources=[
            'torch_utils/ops/filtered_lrelu.cpp',
            'torch_utils/ops/filtered_lrelu_wr.cu',
            'torch_utils/ops/filtered_lrelu_rd.cu',
            'torch_utils/ops/filtered_lrelu_ns.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '--allow-unsupported-compiler'],
        },
    ),
    CUDAExtension(
        'upfirdn2d_plugin',
        sources=[
            'torch_utils/ops/upfirdn2d.cpp',
            'torch_utils/ops/upfirdn2d.cu',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '--allow-unsupported-compiler'],
        },
    ),
]

setup(
    name='stylegan3-ops',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
)
