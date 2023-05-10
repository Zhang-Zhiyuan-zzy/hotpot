"""
python v3.7.9
@Project: hotpot
@File   : gcmc.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/30
@Time   : 22:19
"""
import os
import os.path as osp
from os.path import join as ptj
from typing import *
import src

dir_force_field = osp.abspath(ptj(src.data_root, 'force_field'))


class GCMC:
    """ Performing Grand Canonical Monte Carlo Simulation by LAMMPS """
    def __init__(self, mol, ff, *guests, **kwargs):
        self._data = {
            'mol': mol,
            'guests': guests,
            'ff': self._force_field_file(ff)
        }
        self._data.update(kwargs)

    @staticmethod
    def _force_field_file(ff: Union[str, os.PathLike]):
        """ Retrieve the exact file path of force file """
        # Check the arguments
        # Determine the path of force field.
        if isinstance(ff, os.PathLike):
            pff = str(ff)
        elif osp.exists(ff):
            pff = ff
        else:
            pff = ptj(dir_force_field, ff)

        # Make sure the existence of force field file
        if not osp.exists(pff):
            raise FileNotFoundError('the given force field file is not found!')

        pff = osp.abspath(pff)

        return pff

    def guest_type_map(self):
        pass

    def frame_type_map(self):
        return self.mol.lmp.type_map()

    @property
    def mol(self):
        return self._data.get('mol')

    @property
    def guests(self):
        return self._data.get('guests')

    def group_atoms(self):
        """"""
        frame_atoms = self.mol.num_atoms
        guest_atoms = [g.num_atoms for g in self.guests]

    def run(self):
        self.mol.lmp_setup()
        lmp = self.mol.lmp

        lmp('units real')
        lmp('dimension 3')
        lmp('atom_style full')

        # lmp('read_data /home/zz1/qyq/main.data group frame extra/atom/types 1')
        lmp.read_main_data()
        lmp('mass 2 253.808946')

        lmp('molecule iodine /home/zz1/qyq/lmpmol toff 1')
        lmp('group Igas empty')

        lmp('pair_style lj/cut 5.5')
        lmp(f'pair_coeff 1 1 {52.83 * 0.001987} 3.43')
        lmp(f'pair_coeff 2 2 {550.0 * 0.001987} 4.982')

        lmp('fix stand frame setforce 0.0 0.0 0.0')

        lmp(
            'fix gcmc Igas gcmc 1 $((20+abs(count(Igas)-20))/2) $((20+abs(count(Igas)-20))/2)'
            ' 0 354568 298.15 100 10 mol iodine pressure 52.5573 tfac_insert 15'
        )

        lmp(
            f'fix out all print 1 "$(step),$(count(Igas)),$(mass(Igas)),$(mass(Igas)/mass(frame))" '
            f'file /home/zz1/qyq/I2.csv '
            'title "step,I2_count,I2_mass,uptake(g/g)" screen no'
        )
        lmp('variable uptake equal mass(Igas)/mass(frame)')
        lmp('fix ave all ave/time 1 50 50 v_uptake file /home/zz1/qyq/ave')

        lmp('thermo_style    custom step temp pe etotal press vol density')
        lmp('thermo          1000')
        lmp('compute_modify thermo_temp dynamic/dof yes')

        lmp(f'dump mq all xyz 100 /home/zz1/qyq/I2.xyz')
        lmp(f'dump_modify mq element C, I')

        lmp('timestep 0.0001')
        lmp('run 15000')

