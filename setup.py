import os
from pathlib import Path
import typing

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

SETUP_DIR = Path(__file__).resolve().parent
SRC_DIR = SETUP_DIR / "src"


def make_single_src_extension(src_path: Path, *args, **kwargs) -> Extension:
    ext_rel_path = src_path.relative_to(SRC_DIR)
    ext_name = str(ext_rel_path.with_suffix("")).replace(os.path.sep, ".")
    return Extension(
        ext_name,
        [str(src_path.relative_to(SETUP_DIR))],
        *args,
        **kwargs
    )


src_files = [
    SRC_DIR / "jam_echo_gen/utils.pyx",
    SRC_DIR / "jam_echo_gen/echo_gen.pyx"
]
pyx_src_files = []
c_src_files = []
cpp_src_files = []
for i in range(len(src_files)):
    p = Path(src_files[i])
    if p.suffix == ".pyx":
        if p.is_file():
            pyx_src_files.append(p)
            continue

        # try to find c source
        p = p.with_suffix(".c")
        if p.is_file():
            c_src_files.append(p)
            continue

        # try to find cpp source
        p = p.with_suffix(".cpp")
        if p.is_file():
            cpp_src_files.append(p)
            continue

        # no source found
        raise ValueError(f"No file found for {src_files[i]}")

    else:
        if not p.is_file():
            raise ValueError(f"No file found for {src_files[i]}")

        if p.suffix == ".c":
            c_src_files.append(p)
        elif p.suffix == ".cpp":
            cpp_src_files.append(p)
        else:
            raise ValueError(f"Unknown source file type: {src_files[i]}")

np_include_dir = numpy.get_include()
pyx_extensions = [
    make_single_src_extension(
        src_file,
        include_dirs=[np_include_dir],
        extra_compile_args=["-O3"]
    ) for src_file in pyx_src_files
]
c_extensions = [
    make_single_src_extension(
        src_file,
        include_dirs=[np_include_dir],
        extra_compile_args=["-O3"]
    ) for src_file in c_src_files
]
cpp_extensions = [
    make_single_src_extension(
        src_file,
        include_dirs=[np_include_dir],
        extra_compile_args=["-O3"]
    ) for src_file in cpp_src_files
]

setup(
    ext_modules=(
        cythonize(pyx_extensions) +
        cpp_extensions +
        c_extensions
    )
)
