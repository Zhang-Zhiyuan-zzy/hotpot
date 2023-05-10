"""
python v3.7.9
@Project: hotpot
@File   : gcmc.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/30
@Time   : 22:19
"""
import json
import os
import os.path as osp
import random
from os.path import join as ptj
from typing import *
import src

dir_force_field = osp.abspath(ptj(src.data_root, 'force_field'))


class LjGCMC:
    """ Performing Grand Canonical Monte Carlo Simulation by LAMMPS, based on LJ potential """
    def __init__(self, frame, ff, *guests, **kwargs):
        self._data = {
            'frame': frame,
            'guests': guests,
            'pff': self._force_field_file(ff)  # the path of force field
        }
        self._data.update(kwargs)
        self._load_lj()  # Load force field

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

    def _load_lj(self):
        """ load force file from file """
        self._data['lj'] = json.load(open(self._data['pff']))

    def guest_type_map(self):
        """
        Find out all unique atomic symbol and pseudo atomic symbol in every guest, and then:
            1) Tag each unique atomic symbol with the LAMMPS atom types (a sorted int number)
            2) Extract the first the atom in all the atoms with same atomic symbol or pseudo atomic symbol as
             the representation for all of them
        Returns:
            Dict(int, Union(atom, pseudo_atom))
        """
        num_atom_frame = len(self.frame_type_map())
        guest_type_map = {}
        for guest in self.guests:
            # Extract guest atoms
            atomic_symbols = guest.atomic_symbols
            uni_sym = set(atomic_symbols)  # unique atomic symbols
            for sym in uni_sym:
                # Retrieve the first unique atom and label it with the atom type in LAMMPS
                first_atom_idx = atomic_symbols.index(sym)
                # The atom type to represent for the atomic symbol equals
                #   the number of atom in the frame plus the number of atom types
                #   have assign to other guest atoms or pseudo atoms
                guest_type_map[len(guest_type_map)+1+num_atom_frame] = guest.atoms[first_atom_idx]

            # Extract guest pseudo atoms
            patom_symbols = [pa.symbol for pa in guest.pseudo_atoms]
            uni_psym = set(patom_symbols)
            for psym in uni_psym:
                # Retrieve the first unique pseudo atom and label it with the atom type in LAMMPS
                first_patom_idx = patom_symbols.index(psym)
                guest_type_map[len(guest_type_map)+1+num_atom_frame] = guest.pseudo_atoms[first_patom_idx]

        return guest_type_map

    @property
    def lj(self):
        """ Retrieve the lj potential saved in the class """
        return self._data.get('lj')

    def frame_type_map(self):
        return self.frame.lmp.data_to_labelmap(self.frame.dump('lmpdat'))

    @property
    def frame(self):
        return self._data.get('frame')

    @property
    def guests(self):
        return self._data.get('guests')

    def run(self):
        self.frame.lmp_setup()
        lmp = self.frame.lmp

        # Initialization
        lmp('units real')
        lmp('dimension 3')
        lmp('atom_style full')

        # lmp('read_data /home/zz1/qyq/main.data group frame extra/atom/types 1')
        lmp.read_main_data(extra_atoms=len(self.guest_type_map()))

        self._set_guest_mass(lmp)

        self._load_guest_mol(lmp)

        self._set_lj_coeff(lmp)

        # Keep the atom in frameworks in immobility
        lmp('fix stand frame setforce 0.0 0.0 0.0')

        self._fix_gcmc(lmp)

        self._print_uptake(lmp)

        # lmp('variable uptake equal mass(Igas)/mass(frame)')
        # lmp('fix ave all ave/time 1 50 50 v_uptake file /home/zz1/qyq/ave')

        lmp('thermo_style    custom step temp pe etotal press vol density')
        lmp('thermo          1000')
        lmp('compute_modify thermo_temp dynamic/dof yes')

        lmp(f'dump mq all xyz 100 /home/zz1/qyq/{self.frame.identifier}.xyz')
        lmp(f'dump_modify mq element ' + ', '.join(self.type_symbol_map()))

        lmp('timestep 0.0001')
        lmp('run 15000')

    def type_symbol_map(self):
        symbols = []
        for atom_type, symbol in self.frame_type_map().items():
            symbols.append(symbol)

        for atom_type, symbol in self.guest_type_map().items():
            symbols.append(symbol)

        return symbols

    def _fix_gcmc(self, lmp):
        """"""
        T = self._data.get('T')  # temperature

        for i, guest in enumerate(self.guests, 1):
            guest.thermo_init(T=T)
            phi = guest.thermo.phi  # the fugacity coefficient
            lmp(
                f'fix gcmc{i} gg{i} gcmc 1 $((20+abs(count(Igas)-20))/2) $((20+abs(count(Igas)-20))/2)'
                f' 0 {random.randint(10000, 999999)} {T} 0 12.5 mol guest{i} fugacity_coeff {phi}'
            )

    def _load_guest_mol(self, lmp):
        """"""
        num_ft = len(self.frame_type_map())  # the number of atom types in framework
        num_bgt = 0  # the number of types before current guest
        for i, guest in enumerate(self.guests, 1):
            p_guest_mol = osp.abspath(f'guest{i}')
            lmp(f'molecule guest{i} {p_guest_mol} toff {num_ft+num_bgt}')
            lmp(f'group gg{i} empty')  # Empty group to contain the molecule atoms insert by GCMC

            num_bgt += len(set(guest.atomic_symbols)) + len(set(pa.symbol for pa in guest.pseudo_atoms))

    def _print_uptake(self, lmp):
        """ print the guest uptake with guest(g)/framework(g) into file """
        data = '"' + '$(step),' + ','.join(f'$(mass(gg{i})/mass(frame))' for i, _ in enumerate(self.guests, 1)) + '"'
        title = '"' + 'step,' + ','.join(f'{g.smiles}(g/g)' for i, g in enumerate(self.guests, 1)) + '"'

        p_file = osp.abspath(self._data.get('print_file', 'step_uptake.csv'))

        cmd = f'fix uptake all print 1 {data} title {title} file {p_file}'

        lmp(cmd)

    def _set_guest_mass(self, lmp: "HpLammps"):
        for atom_type, atom in self.guest_type_map().items():
            lmp(f'{atom_type} {atom.mass}')

    def _set_lj_coeff(self, lmp: 'HpLammps'):
        # the LJ cut off
        lmp(f'pair_style lj/cut {self._data.get("cut", 12.5)}')

        # LJ coefficient for framework atoms
        for atom_type, atomic_symbol in self.frame_type_map().items():
            eps = self.lj[atomic_symbol]["epsilon"]
            sigma = self.lj[atomic_symbol]["sigma"]
            lmp(f'pair_coeff {atom_type} {atom_type} {eps * 0.001987} {sigma}')

        # LJ coefficient for guest atoms
        # atom: the atom or pseudo atom corresponding to the atom_type
        for atom_type, atom in self.guest_type_map().items():
            eps = self.lj[atom.symbol]["epsilon"]
            sigma = self.lj[atom.symbol]["sigma"]
            lmp(f'pair_coeff {atom_type} {atom_type} {eps * 0.001987} {sigma}')


import src.cheminfo as ci
from src.tanks.lmp.base import HpLammps
