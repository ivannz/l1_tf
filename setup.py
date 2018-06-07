"""Setup script for the L1_tf wrapper."""
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

import numpy


extensions = [
    Extension("l1tf._l1tf",
              sources=["l1tf/_l1tf.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["blas", "lapack"],
              extra_compile_args=["-std=c99", "-O3", "-Ofast"])
]

# python setup.py build_ext [--inplace]
setup(
    name="l1tf",
    version="0.1.6",
    description="""A python wrapper for L1 trend filtering via primal-dual """
                """algorithm by Kwangmoo Koh, Seung-Jean Kim, and Stephen """
                """Boyd (http://stanford.edu/~boyd/l1_tf/).""",
    author="Ivan Nazarov",
    license='GNU',
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=["numpy", "cython"],
)
