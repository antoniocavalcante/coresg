from setuptools import setup
from Cython.Build import cythonize

setup(
    name='rng',
    ext_modules=cythonize("*.pyx"),
    zip_safe=False,
)