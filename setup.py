"""Setup script for the L1_tf wrapper."""
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from pip.req import parse_requirements
from pip.download import PipSession

import numpy

install_reqs = parse_requirements("requirements.txt", session=PipSession())

# python setup.py build_ext [--inplace]
setup(
    name="l1tf",
    ext_modules=cythonize([Extension("l1tf/_l1tf", ["l1tf/_l1tf.pyx"],
                                     include_dirs=[numpy.get_include()],
                                     libraries=["blas", "lapack", "m"]),]),
    cmdclass={"build_ext": build_ext},
    packages=["l1tf",],
    author='Ivan Nazarov',
    version='0.1.5',
    description="""A python wrapper for L1 trend filtering via primal-dual algorithm """
                """by Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd """
                """(http://stanford.edu/~boyd/l1_tf/)""",
    install_requires=[str(ir.req) for ir in install_reqs],
)
