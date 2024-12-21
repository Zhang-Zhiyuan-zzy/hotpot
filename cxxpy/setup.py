"""
python v3.9.0
@Project: hotpot
@File   : setup.py
@Auther : Zhiyuan Zhang
@Data   : 2024/12/17
@Time   : 15:03
"""
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'cxxconvert',  # The name of the module
        ['convert.cpp'],  # Your C++ source file
        include_dirs=[pybind11.get_include()],  # Add include path for pybind11
        language='c++',
        extra_compile_args=['-std=c++17'],
    ),
]

setup(
    name='cxxconvert',
    ext_modules=ext_modules,
    zip_safe=False,
)