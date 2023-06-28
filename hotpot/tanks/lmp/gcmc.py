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
import hotpot

dir_force_field = osp.abspath(ptj(hotpot.data_root, 'force_field'))


class LjGCMC:
    """ Performing Grand Canonical Monte Carlo Simulation by LAMMPS, based on LJ potential """
    def __init__(
            self, frame: 'ci.Molecule', ff: Union[str, os.PathLike], *guests: 'ci.Molecule',
            work_dir: Union[str, os.PathLike], **kwargs
    ):
        self._data = {
            'frame': frame,
            'guests': guests,
            'work_dir': work_dir,
            'pff': self._force_field_file(ff)  # the path of force field
        }

        # Preprocessing data
        self._data.update(kwargs)
        self._load_lj()  # Load force field
        self._extract_type_map()

    def _extract_type_map(self):
        def ext_tm(mol: 'ci.Molecule', which):
            atom_types_map = self._data.setdefault('atom_types_map', {})
            atom_types = self._data.setdefault('atom_types', {})
            list_atom_types = atom_types.setdefault(which, [])
            for atom in mol.all_atoms_with_unique_symbol:
                atom_type = len(atom_types_map) + 1
                atom_types_map[atom_type] = atom
                list_atom_types.append(atom_type)

        ext_tm(self.frame, 'frame')
        for guest in self.guests:
            ext_tm(guest, 'guest')

    def _fix_gcmc(self, lmp):
        """"""
        T = self._data.get('T')  # temperature
        P = self._data.get('P', 1.0)  # the ratio pressure to saturation pressure

        for i, guest in enumerate(self.guests, 1):
            guest.thermo_init(T=T)
            phi_g = guest.thermo.phi_g  # the fugacity coefficient
            P_sat = guest.thermo.Psat
            lmp(
                f'fix gcmc{i} gg{i} gcmc 1 $((20+abs(count(gg{i})-20))/2) $((20+abs(count(gg{i})-20))/2)'
                f' 0 {random.randint(10000, 999999)} {T} 0 12.5 mol guest{i} fugacity_coeff {phi_g} pressure {P_sat * P}'
            )

    @staticmethod
    def _force_field_file(ff: Union[str, os.PathLike]):
        """ Retrieve the exact file path of force file """

        # If the force filed not be given, use the UFF as default.
        if not ff:
            ff = 'UFF/LJ.json'

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

    def _print_uptake(self, lmp):
        """ print the guest uptake with guest(g)/framework(g) into file """
        data = '"' + '$(step),' + ','.join(f'$(mass(gg{i})/mass(frame))' for i, _ in enumerate(self.guests, 1)) + '"'
        title = '"' + 'step,' + ','.join(f'{g.smiles}(g/g)' for i, g in enumerate(self.guests, 1)) + '"'

        p_file = osp.abspath(ptj(self.work_dir, 'step_uptake.csv'))

        cmd = f'fix uptake all print 1 {data} title {title} file {p_file} screen no'

        lmp(cmd)

    def _load_guest_mol(self, lmp):
        """"""
        num_ft = len(self.frame_type_map)
        num_fbt = len(self.frame.unique_bonds)  # the number of atom types in framework
        num_gat = 0  # the number of atom types before current guest
        num_gbt = 0  # the number of bond types before current guest
        for i, guest in enumerate(self.guests, 1):
            if self.work_dir:
                p_guest_mol = osp.abspath(ptj(self.work_dir, f'guest{i}'))
            else:
                p_guest_mol = osp.abspath(f'guest{i}')

            guest.writefile('lmpmol', p_guest_mol, retrieve_script=True, atom_style='metal')

            lmp(f'molecule guest{i} {p_guest_mol} toff {num_ft+num_gat} boff {num_fbt + num_gbt}')
            lmp(f'group gg{i} empty')  # Empty group to contain the molecule atoms insert by GCMC

            num_gat += len(set(guest.atomic_symbols)) + len(set(pa.symbol for pa in guest.pseudo_atoms))
            num_fbt += len(guest.unique_bond_pairs)

    def _load_lj(self):
        """ load force file from file """
        self._data['lj'] = json.load(open(self._data['pff']))

    def _set_guest_mass(self, lmp: "HpLammps"):
        for atom_type, atom in self.guest_type_map.items():
            lmp(f'mass {atom_type} {atom.mass}  # {atom.symbol}')

    def _set_lj_coeff(self, lmp: 'HpLammps'):
        # the LJ cut off
        lmp(f'pair_style lj/cut {self._data.get("cut", 12.5)}')

        # LJ coefficient for guest atoms
        # atom: the atom or pseudo atom corresponding to the atom_type
        # TODO: the coefficient of epsilon
        for atom_type, atom in self.atom_types_map.items():
            try:
                eps = self.lj[atom.symbol]["epsilon"]
                sigma = self.lj[atom.symbol]["sigma"]
            except KeyError:
                assert isinstance(atom, ci.PseudoAtom)
                eps = atom.epsilon
                sigma = atom.sigma

            lmp(f'pair_coeff {atom_type} {atom_type} {eps * 0.001987} {sigma}')

    @property
    def atom_types_map(self) -> Dict[int, Union['ci.Atom', 'ci.PseudoAtom']]:
        return self._data.get('atom_types_map')

    @property
    def frame_atom_types(self) -> List[int]:
        return self._data.get('atom_types')['frame']

    @property
    def guest_atom_types(self) -> List[int]:
        return self._data.get('atom_types')['guest']

    @property
    def frame(self):
        return self._data.get('frame')

    @property
    def frame_type_map(self) -> Dict[int, Union['ci.Atom', 'ci.PseudoAtom']]:
        """ Return the dict mapping from type_numbers to Atom or PseudoAtom objects in framework """
        return {t_num: atom for t_num, atom in self.atom_types_map.items() if t_num in self.frame_atom_types}

    @property
    def guest_type_map(self) -> Dict[int, Union['ci.Atom', 'ci.PseudoAtom']]:
        """ Return the dict mapping from type_numbers to Atom or PseudoAtom objects in guests """
        return {t_num: atom for t_num, atom in self.atom_types_map.items() if t_num in self.guest_atom_types}

    @property
    def guests(self):
        return self._data.get('guests')

    @property
    def lj(self):
        """ Retrieve the lj potential saved in the class """
        return self._data.get('lj')

    @property
    def num_atom_types(self):
        return len(self.atom_types_map)

    @property
    def num_frame_atom_types(self) -> int:
        return len(self.frame_atom_types)

    @property
    def num_guest_atom_types(self) -> int:
        return len(self.guest_atom_types)

    def run(self):
        self.frame.lmp_setup(work_dir=self.work_dir)
        lmp = self.frame.lmp

        # Initialization
        lmp('units real')
        lmp('dimension 3')
        lmp('atom_style full')

        # lmp('read_data /home/zz1/qyq/main.data group frame extra/atom/types 1')
        lmp.read_main_data(
            extra_atom_types=self.num_guest_atom_types,
            extra_bond_types=sum([len(g.unique_bonds) for g in self.guests]),
            group='frame'
        )

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

        dir_dump = osp.abspath(ptj(self.work_dir, "dump.xyz"))
        lmp(f'dump mq all xyz 100 {dir_dump}')
        lmp(f'dump_modify mq element ' + ', '.join(self.type_symbol_map))

        lmp('timestep 0.0001')
        lmp('run 15000')

    @property
    def type_symbol_map(self) -> List[str]:
        return [atom.symbol for atom in self.atom_types_map.values()]

    @property
    def work_dir(self):
        return self._data.get('work_dir', os.getcwd())


import hotpot.cheminfo as ci
from hotpot.tanks.lmp.base import HpLammps
