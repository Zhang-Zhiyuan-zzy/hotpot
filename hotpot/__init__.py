"""
python v3.7.9
@Project: hotpot
@File   : __init__.py.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/14
@Time   : 4:06
"""
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

hp_root = os.path.abspath(os.path.dirname(__file__))
data_root = os.path.abspath(os.path.join(hp_root, 'data'))

# Reading User custom Settings
try:
    settings = json.load(open(os.path.join(data_root, "settings.json")))
except FileNotFoundError:
    settings = {}

# Adding paths into local variables
locals().update(settings.get('paths', {}))


# function to configure settings
def settings_config(new_settings: dict):
    json.dump(new_settings, open(os.path.join(data_root, "settings.json"), 'w'), indent=True)


from hotpot.cheminfo import Molecule
from hotpot.bundle import MolBundle

__version__ = '0.3.4.0'
