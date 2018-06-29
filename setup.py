"""Setup script for the L1_tf wrapper."""
import os
from os.path import join

import numpy
from numpy.distutils.system_info import get_info

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize


blas_info = get_info('blas_opt', 0)
cblas_libs = blas_info.pop('libraries', [])
if os.name == 'posix':
    cblas_libs.append('m')

sources = [join("l1tf", "_l1tf.pyx")]
depends = []

include_dirs = [numpy.get_include()]
include_dirs.extend(blas_info.pop("include_dirs", []))

extra_compile_args = blas_info.pop("extra_compile_args", [])
extra_compile_args.extend(["-std=c99", "-O3", "-Ofast"])

extensions = [
    Extension("l1tf._l1tf", sources=sources, libraries=cblas_libs,
              include_dirs=include_dirs, extra_compile_args=extra_compile_args,
              depends=depends, **blas_info)
]

# python setup.py build_ext [--inplace]
setup(
    name="l1tf",
    version="0.1.6",
    description="""A python wrapper for L1 trend filtering via primal-dual """
                """algorithm by Kwangmoo Koh, Seung-Jean Kim, and Stephen """
                """Boyd (http://stanford.edu/~boyd/l1_tf/).""",
    author="Ivan Nazarov",
    license="GNU",
    packages=find_packages(),
    ext_modules=cythonize(extensions, quiet=True, nthreads=4),
    install_requires=["numpy", "cython"],
)
