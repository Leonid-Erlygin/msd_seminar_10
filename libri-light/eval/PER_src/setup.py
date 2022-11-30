# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules=cythonize("per_operator.pyx"),
    include_dirs=[numpy.get_include()]
)
