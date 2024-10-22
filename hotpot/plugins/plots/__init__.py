"""
python v3.9.0
@Project: hotpot
@File   : __init__.py
@Auther : Zhiyuan Zhang
@Data   : 2024/10/9
@Time   : 12:54
"""
import os
import sys
import shutil
import matplotlib as mpl

from .plotter import SciPlotter
from .plots import *


def refresh_cache():
    cache_dir = mpl.get_cachedir()
    if cache_dir is not None and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def init_fonts():
    """ For linux system, load the Arial as the default font """
    import matplotlib.font_manager as fm
    if sys.platform.startswith('linux'):
        if fm.findfont('Arial'):
            refresh_cache()
        else:
            print(UserWarning('Arial font is not installed, please install by system command, '
                              'like: `apt install ttf-mscorefonts-installer`'))
init_fonts()

