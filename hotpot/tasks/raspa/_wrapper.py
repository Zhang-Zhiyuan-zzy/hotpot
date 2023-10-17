"""
python v3.9.0
@Project: hotpot
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2023/7/27
@Time   : 9:54
"""
import os
import json
import shutil
from typing import Union
from pathlib import Path

import pandas as pd
from openbabel import pybel as pb

import hotpot
from hotpot import settings, Molecule
from hotpot.tasks.raspa import _core
from ._results import RaspaParser

_raspa_root = settings.get("paths", {}).get('raspa_root')


def make_forcefield(path, **kwargs):
    """
     The path variable refers to the path of the json file
     Keyword Args: 'general_rule_for' and 'general_rule_tailcorrections'
     Currently, only the conversion of json files into force_field_mixing_rules.def is supported
    """
    with open(path, 'r') as file:
        json_data = json.load(file)
    num_defined_interaction = len(json_data)
    # Convert nested dict into appropriately dataframe
    json_df = pd.DataFrame(json_data).T
    json_df['symbol'] = json_df.index
    json_df = json_df.reset_index(drop=True)
    new_order = ['symbol', 'epsilon', 'sigma']
    new_json_df = json_df[new_order]

    # Determine whether kwargs are provided, if not, use the default value
    if 'general_rule_for' in kwargs:
        general_rule_for = kwargs['general_rule_for']
    else:
        general_rule_for = 'shifted'

    if 'general_rule_tailcorrections' in kwargs:
        general_rule_tailcorrections = kwargs['general_rule_tailcorrections']
    else:
        general_rule_tailcorrections = 'no'

    # Script to generate force field file
    script = f'# general rule for shifted vs truncated\n{general_rule_for}\n# general rule tailcorrections\n{general_rule_tailcorrections}\n'
    script += f'# number of defined interactions\n{num_defined_interaction}\n'
    script += '# type interaction, parameters.    IMPORTANT: define shortest matches first, so that more specific ones overwrites these\n'
    # Add suffix '_' to the first 102 fixed elements
    for i, val in enumerate(new_json_df.values):
        if i < 102:
            line_str = f'{val[0]}_\t\tlennard-jones\t{val[1]}\t{val[2]}\n'
        else:
            line_str = f'{val[0]}\t\tlennard-jones\t{val[1]}\t{val[2]}\n'
        script += line_str
    # Save the script of the force field file into the RASPA force field folder
    save_dir = os.path.dirname(path)
    save_path = os.path.join(save_dir, 'force_field_mixing_rules.def')
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(script)

    return script


class RASPA:
    """ A python wrapper for running a single RASPA calculation """
    def __init__(
            self,
            forcefield: str = "UFF",
            raspa_root: Union[str, os.PathLike] = None,
            guest_dir_name: Union[str, os.PathLike] = "Hotpot",
            in_test: bool = False
    ):
        """ Initialization """
        if raspa_root:
            self.raspa_root = Path(raspa_root)
        elif _raspa_root:
            self.raspa_root = Path(_raspa_root)
        else:
            raise ValueError('the arg raspa_root is not specified !!!')

        self.data_root = Path(hotpot.data_root)

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

        self.in_test = in_test

        self._check_force_field()

    def _check_force_field(self):
        """
        Check if the force field has been properly defined
        Force field files can be converted from json files and the UFF files can be copied from the data folder
        Enriching UFF force field element categories from data files
        """
        def enrich_uff_pseudo_ele_types():
            """ Enrich UFF pseudo_atoms file element types from data folder"""
            with open(pseudo_ff_path, 'r') as pseudo_file:
                pseudo_list = pseudo_file.readlines()
            with open(src_pseudo_ff_path, 'r') as src_pseudo_file:
                src_pseudo_list = src_pseudo_file.readlines()

            # Summarize the element types in pseudo file in UFF
            ele_list = []
            for i, line in enumerate(pseudo_list):
                if i > 2:
                    ele = line.split()[0]
                    ele_list.append(ele)

            # Copy additional element types from the data folder into UFF pseudo file
            supply_script_list = []
            for i, src_line in enumerate(src_pseudo_list):
                if i > 2:
                    src_ele = src_line.split()[0]
                    if src_ele not in ele_list:
                        supply_script_list.append(src_line)
            if supply_script_list:
                # Update the total number of element types in pseudo file
                init_pseudo_num = int(pseudo_list[1])
                added_pseudo_num = init_pseudo_num + len(supply_script_list)
                pseudo_list[1] = str(added_pseudo_num) + '\n'
                new_pseudo_list = pseudo_list + supply_script_list
                with open(pseudo_ff_path, 'w') as writer:
                    writer.writelines(new_pseudo_list)

        def enrich_uff_mix_ele_types():
            """ Enrich UFF force_field_mixing_rules file element types from data folder"""
            with open(mix_ff_path, 'r') as mix_file:
                mix_list = mix_file.readlines()
            with open(src_mix_ff_path, 'r') as mix_src_file:
                mix_src_list = mix_src_file.readlines()
            # Summarize the element types in mixing file in UFF
            mix_ele_list = []
            for i, mix_line in enumerate(mix_list):
                if i > 6:
                    if 'general mixing rule for Lennard-Jones' in mix_line:
                        break
                    else:
                        mix_ele = mix_line.split()[0]
                        mix_ele_list.append(mix_ele)

            # Copy additional element types from the data folder into UFF mixing file
            mix_supply_script_list = []
            for i, mix_src_line in enumerate(mix_src_list):
                if i > 6:
                    if 'general mixing rule for Lennard-Jones' in mix_src_line:
                        break
                    else:
                        mix_src_ele = mix_src_line.split()[0]
                        if mix_src_ele not in mix_ele_list:
                            mix_supply_script_list.append(mix_src_line)
            if mix_supply_script_list:
                # Update the total number of element types in mixing file
                init_mix_num = int(mix_list[5])
                added_mix_num = init_mix_num + len(mix_supply_script_list)
                mix_list[5] = str(added_mix_num) + '\n'

                # the general mixing rules should always be in the last two lines of mixing file
                mix_rule_list = mix_list[-2:]
                cut_first_mix_list = mix_list[:-2]
                cut_second_mix_list = cut_first_mix_list + mix_supply_script_list
                new_mix_list = cut_second_mix_list + mix_rule_list
                with open(mix_ff_path, 'w') as mix_file:
                    mix_file.writelines(new_mix_list)

        force_field_dir = self.raspa_root.joinpath("share", "raspa", "forcefield", self.forcefield)
        mix_ff_path = os.path.join(force_field_dir, 'force_field_mixing_rules.def')
        pseudo_ff_path = os.path.join(force_field_dir, 'pseudo_atoms.def')

        if self.forcefield == "UFF":
            src_pseudo_ff_path = self.data_root.joinpath("force_field", "UFF", "pseudo_atoms.def")
            src_mix_ff_path = self.data_root.joinpath("force_field", "UFF", "force_field_mixing_rules.def")
            if not os.path.exists(force_field_dir):      # Make sure the RASPA forcefield folder exists
                os.mkdir(force_field_dir)
            if not os.path.exists(mix_ff_path):
                shutil.copy2(src_mix_ff_path, mix_ff_path)
            if not os.path.exists(pseudo_ff_path):
                shutil.copy2(src_pseudo_ff_path, pseudo_ff_path)

            # Enrich UFF force field element types
            enrich_uff_pseudo_ele_types()
            enrich_uff_mix_ele_types()

        else:
            if not os.path.exists(force_field_dir):
                raise FileNotFoundError(f"the force field files of {str(self.forcefield)} are not found!")

            else:
                if not os.path.exists(pseudo_ff_path):
                    raise FileNotFoundError(f"the pseudo atoms file of {str(self.forcefield)} is not found!")
                if not os.path.exists(mix_ff_path):
                    json_path = None
                    for file in os.listdir(force_field_dir):
                        if '.json' in file:
                            json_path = os.path.join(force_field_dir, file)
                            make_forcefield(json_path)
                    if not json_path:
                        raise FileNotFoundError(f"the force field mixing rules file of {str(self.forcefield)} is not found!")

    def _guests_to_mol_files(self, guests: tuple[Union[str, Molecule]], *args, **kwargs) -> list[str]:
        """
        Convert the hotpot.Molecule to raspa Molecule file or copy raspa Molecule file from the data folder
        Args:
            guests(Molecule):

        Returns:
            string script with the
        """
        guest_dir = self.raspa_root.joinpath("share", "raspa", "molecules", self.guest_dir_name)
        if not os.path.exists(guest_dir):     # Make sure RASPA molecule folder Hotpot exists
            os.mkdir(guest_dir)

        guest_names = []
        for guest in guests:
            if isinstance(guest, str):
                # When a guest file name is given, checking whether the defined guest file exist
                guest_path = guest_dir.joinpath(f"{guest}.def")
                if not guest_path.exists():
                    src_guest_path = self.data_root.joinpath("raspa_mol", f"{guest}.def")

                    if os.path.exists(src_guest_path):
                        shutil.copy2(src_guest_path, guest_path)
                    else:
                        raise FileNotFoundError(f"the guest file {str(guest_path)} is not found!")

                guest_names.append(guest)

            else:  # When a Molecule object is given, convert the Molecule to be the raspa_mol text and write to disk.
                guest_name = guest.identifier
                guest_path = guest_dir.joinpath(f"{guest_name}.def")

                if guest_path.exists():
                    if self.in_test:
                        option = "UseOld"

                    else:
                        message = \
                            f"""the definition file {str(guest_path)} has existed, which action is your want?
                            It should be care for the option 'Override', which will remove the exist defined
                            raspa_mol file!! select from: 
                            [Quit/UseOld/Override:]
                            """
                        while (option := input(RuntimeWarning(message))) not in ["Quit", "UseOld", "Override"]:
                            pass

                    if option == "Quit":
                        raise FileExistsError('the guest definition file has existed')
                    elif option == "Override":
                        guest.writefile('raspa_mol', guest_path, *args, **kwargs)

                else:
                    src_guest_path = self.data_root.joinpath("raspa_mol", f"{guest_name}.def")

                    if os.path.exists(src_guest_path):
                        shutil.copy2(src_guest_path, guest_path)
                    else:
                        guest.writefile('raspa_mol', guest_path, *args, **kwargs)

                guest_names.append(guest_name)

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
