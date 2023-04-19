# Copyright 2022 PIERRE LASSALLE
# All rights reserved

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
from pybind11.setup_helpers import Pybind11Extension, build_ext

extensions = [
    Extension("demogreen.cyuniform", ["src/demogreen/cylib/demogreen.pyx"]),
    Pybind11Extension("pbuniform", ["src/demogreen/pybindlib/pybind_uniform_lib.cpp"])
]

compiler_directives = { "language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

setup(
    ext_modules=extensions,
    install_requires=install_requires,
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    cmdclass={"build_ext": build_ext}
)