from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Default architectures for broad RTX support including future cards like RTX5090
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.0;7.5;8.0;8.6;8.9;9.0')

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
