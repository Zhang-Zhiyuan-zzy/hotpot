import os
import shutil
import h5py
import os.path as osp
from typing import Generator
from glob import glob

from hotpot import Molecule
from .. import utils
from .pyg_data import LnQM_Dataset


ulr = 'https://zenodo.org/api/records/10406124/files-archive'

_module_root = osp.dirname(__file__)
_data_dir = osp.join(_module_root, 'data')

_download_file_name = 'files-archive'
_download_file_path = osp.join(_data_dir, _download_file_name)

_h5file = osp.join(_data_dir, 'lnqm.h5')
_geometric_dir = osp.join(_data_dir, 'geometries')

_attrs = [
    'ceh', 'charge', 'cn', 'coord', 'dipole', 'dipole_ele',
    'dipole_nuc', 'dipole_tot', 'ec', 'ecnl', 'eemb', 'eeq',
    'energy', 'energy_geoopt', 'etot_geoopt', 'ex', 'exc',
    'gradient', 'hirshfeld_alpha', 'hirshfeld_beta', 'hirshfeld_charges',
    'hirshfeld_spins', 'homo_spin_down', 'homo_spin_up', 'loewdin_charges',
    'loewdin_spins', 'lumo_spin_down', 'lumo_spin_up', 'mayer_bo', 'mayer_pop',
    'mulliken_charges', 'mulliken_spins', 'nel', 'nel_alpha', 'nel_beta',
    'numbers', 'orbital_energies_spin_down', 'orbital_energies_spin_up',
    'polarizabilities', 'q_gfn2', 'rot_const', 'rot_dipole', 'time_geoopt',
    'time_singlepoint', 'trajectory', 'trajectory_energies', 'trajectory_etot',
    'trajectory_gradients', 'uid', 'unpaired_e'
]


class LnQmDataset:
    def __init__(self):
        if not osp.exists(_data_dir):
            os.mkdir(_data_dir)

        # Check whether the files complete.
        # If the file is not complete download and unzip file.
        if not osp.exists(osp.join(_data_dir, 'Done')):
            for file in glob(osp.join(_data_dir, '*')):
                if osp.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)

            utils.download_file(_download_file_path, ulr)
            utils.uncompress_zip(_download_file_path, _data_dir)
            utils.uncompress_zip(osp.join(_data_dir, 'lnqm_geometries.zip'), _data_dir)

            # Remove intermediate files
            os.remove(_download_file_path)
            os.remove(osp.join(_data_dir, 'lnqm_geometries.zip'))

            with open(osp.join(_data_dir, 'Done'), 'w') as f:
                f.write('Done')

    def to_pyg_dataset(self):
        return LnQM_Dataset(_h5file)

    def extract(self) -> Generator["Molecule", None, None]:
        with h5py.File('/home/zzy/proj/bayes/lnQm/lnqm.h5') as f:
            data = f['data']  # 726039
            slices = f['slices']  # 17270

            items = list(data.keys())

            # TODO: Yu TongXin
