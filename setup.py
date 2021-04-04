from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

extensions = [
    Extension(
        "mst",
        ["mst/mst.pyx"],
        extra_compile_args = ["-ffast-math"]
    ),
    Extension(
        "rng",
        ["rng/rng.pyx"],
        extra_compile_args = ["-ffast-math"]
    ),
    Extension(
        "fst",
        ["rng/fair_split_tree.pyx"],
        extra_compile_args = ["-ffast-math"]
    )
]

setup(
    name='hdbscan_python',
    ext_modules=cythonize(extensions, include_path = [numpy.get_include()], annotate=True)
)
