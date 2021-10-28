import os
import glob
import torch

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    exts_dir = os.path.join(this_dir, 'monoex')

    extensions = []
    ext_name = '_ext'
    define_macros = []
    extra_compile_args = {'cxx': []}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = [
            "-DCUDA_HAS_FP16=1",
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        op_files = glob.glob(os.path.join(exts_dir, 'layers/csrc/pytorch', '*'))
        extension = CUDAExtension
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob(os.path.join(exts_dir, 'layers/csrc/pytorch', '*.cpp'))
        extension = CppExtension

    include_dirs = os.path.abspath(os.path.join(exts_dir, 'layers/csrc'))

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_dirs],
        define_macros=define_macros,
        extra_compile_args=extra_compile_args
    )
    extensions.append(ext_ops)

    return extensions


if __name__ == '__main__':
    setup(
        name='monoex',
        version='0.1',
        description='monoex: Private Monocular 3D Object Detection Lib',
        author='excito',
        author_email='excitohe@gmail.com',
        packages=find_packages(exclude=("configs", "tests")),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )
