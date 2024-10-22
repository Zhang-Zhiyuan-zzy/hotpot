"""
python v3.9.0
@Project: hp5
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/6/5
@Time   : 16:55
"""
import os
import sys

from os.path import join, dirname

def version():
    _version = {}
    with open(join(dirname(__file__), '__version__.py'), 'r') as f:
        exec(f.read(), _version)

    return _version['__version__']


# add package root
package_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(package_root)

from .cheminfo.core import Molecule
