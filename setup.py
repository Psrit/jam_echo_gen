import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "jam_echo_gen.utils",
        sources=["src/jam_echo_gen/utils.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "jam_echo_gen.echo_gen",
        sources=["src/jam_echo_gen/echo_gen.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
