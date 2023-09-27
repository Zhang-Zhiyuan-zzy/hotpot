"""
python v3.9.0
@Project: hotpot
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2023/7/27
@Time   : 9:54
"""
import os
import logging
from typing import Union
from pathlib import Path

from openbabel import pybel as pb

from hotpot import settings, Molecule
from hotpot.tasks.raspa import _core
from ._results import RaspaParser

_raspa_root = settings.get("paths", {}).get('raspa_root')


class RASPA:
    """ A python wrapper for running a single RASPA calculation """
    def __init__(
            self,
            forcefield: str = "UFF",
            raspa_root: Union[str, os.PathLike] = None,
            guest_dir_name: Union[str, os.PathLike] = "Hotpot",
    ):
        """ Initialization """
        if raspa_root:
            self.raspa_root = Path(raspa_root)
        elif _raspa_root:
            self.raspa_root = Path(_raspa_root)
        else:
            raise ValueError('the arg raspa_root is not specified !!!')

        self.libraspa_dir = os.path.join(self.raspa_root, "lib")
        self.libraspa_file = next(f for f in os.listdir(self.libraspa_dir) if "libraspa" in f)

        self.core_args = {
            "raspa_dir": str(self.raspa_root),
            "libraspa_dir": self.libraspa_dir,
            "libraspa_file": self.libraspa_file,
        }

        # update the "_core" module namespace
        _core.__dict__.update(self.core_args)

        self.guest_dir_name = guest_dir_name
        self.forcefield = forcefield

        self._check_force_field()

    def _check_force_field(self):
        """ Check if the force field has been properly defined """
        force_field_dir = self.raspa_root.joinpath("share", "raspa", "forcefield", self.forcefield)
        # TODO: Yuqing.

    def _guests_to_mol_files(self, guests: tuple[Union[str, Molecule]], *args, **kwargs) -> list[str]:
        """
        Convert the hotpot.Molecule to raspa Molecule file
        Args:
            guests(Molecule):

        Returns:
            string script with the
        """
        guest_dir = self.raspa_root.joinpath("share", "raspa", "molecules", self.guest_dir_name)
        guest_names = []
        for guest in guests:
            if isinstance(guest, str):
                # When a guest file name is given, checking whether the defined guest file exist
                guest_path = guest_dir.joinpath(f"{guest}.def")
                if not guest_path.exists():
                    raise FileNotFoundError(f"the guest file {str(guest_path)} is not found!")
                guest_names.append(guest)

            else:  # When a Molecule object is given, convert the Molecule to be the raspa_mol text and write to disk.
                guest_path = guest_dir.joinpath(f"{guest.identifier}.def")

                if guest_path.exists():
                    while (option := None) not in ["Quit", "UseOld", "Override"]:
                        option = input(RuntimeWarning(
                            f"the definition file {str(guest_path)} has existed, which action is your want?\n"
                            f"It should be care for the option 'Override', which will remove the exist defined\n"
                            f"raspa_mol file!!\n"
                            f"select from: Quit/UseOld/Override"
                        ))

                    if option == "Quit":
                        raise FileExistsError('the guest definition file has existed')
                    elif option == "Override":
                        guest.writefile('raspa_mol', guest_path, *args, **kwargs)

                    guest_names.append(guest.identifier)

        return guest_names

    def run(
            self, frame: Molecule, *guests: Union[str, Molecule], mol_fractions=None,
            temperature=273.15, pressure=101325, helium_void_fraction=1.0,
            unit_cells=(1, 1, 1), simulation_type="MonteCarlo",
            cycles=10000, init_cycles="auto", input_file_type="cif",
            **kwargs
    ) -> RaspaParser:
        """"""
        # Write the guests molecule file to work_dir
        guest_names = self._guests_to_mol_files(guests, **kwargs)
        assert len(guest_names) == len(guests)

        # When simulating a single component guest
        if len(guests) == 1:
            result = _core.run(
                pb.Molecule(frame.ob_mol), guest_names[0],
                temperature=temperature, pressure=pressure,
                helium_void_fraction=helium_void_fraction,
                unit_cells=unit_cells, simulation_type=simulation_type,
                cycles=cycles, init_cycles=init_cycles,
                input_file_type=input_file_type, forcefield=self.forcefield,
            )

        # When simulating the mixture guests
        else:
            if not mol_fractions:
                mol_fractions = [1.0 / len(guests)] * len(guests)
            else:
                assert len(mol_fractions) == len(guests)

            result = _core.run_mixture(
                pb.Molecule(frame.ob_mol), guest_names, mol_fractions,
                temperature=temperature, pressure=pressure,
                helium_void_fraction=helium_void_fraction,
                unit_cells=unit_cells, simulation_type=simulation_type,
                cycles=cycles, init_cycles=init_cycles,
                input_file_type=input_file_type, forcefield=self.forcefield
            )

        return RaspaParser(result)
