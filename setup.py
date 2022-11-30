# From https://github.com/haopeng-uw/RFA/tree/6d59b1c79b47e78c9ece0ee4c3a97c3fb29488b1

import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from torch.utils.cpp_extension import CUDA_HOME

ext_modules = []

if torch.cuda.is_available() and CUDA_HOME is not None:
    extension = CUDAExtension(
        'rfa_cuda', [
            'models/cuda/rfa_cuda.cpp',
            'models/cuda/forward.cu',
            'models/cuda/backward.cu',
            'models/cuda/causal.cu',
            'models/cuda/calculate_sz.cu',
            'models/cuda/cross.cu',
            'models/cuda/random_project.cu'
        ],
        extra_compile_args={
            'cxx': [
                '-g',
                '-v'],
            'nvcc': [
                '-DCUDA_HAS_FP16=1',
                '-gencode',
                'arch=compute_70,code=sm_70',
                '-use_fast_math'
            ]
        }
    )
    ext_modules.append(extension)
else:
    assert False

setup(
    name='rfa',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
