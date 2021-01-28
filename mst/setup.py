from setuptools import setup
from Cython.Build import cythonize

setup(
    name='mst',
    ext_modules=cythonize("mst.pyx"),
    zip_safe=False,
)