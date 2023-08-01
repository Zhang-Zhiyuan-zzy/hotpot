"""
python v3.9.0
@Project: hotpot
@File   : core
@Auther : Zhiyuan Zhang
@Data   : 2023/7/27
@Time   : 9:54
"""
import os
from ctypes import cdll, c_void_p, c_char_p, c_bool, cast
from typing import Union, Optional
from pathlib import Path
from textwrap import dedent

from ...cheminfo import Molecule


class RASPA:
    """ A python wrapper for running a single RASPA calculation """
    def __init__(
            self,
            raspa_root: Union[str, os.PathLike],
            work_dir: Union[str, os.PathLike],
    ):
        """ Initialization"""
        self.raspa_root = Path(raspa_root)
        self.work_dir = work_dir

    def _make_guest_file(
            self, guest: Molecule, critical_temperature: Optional[float] = None,
            critical_pressure: Optional[float] = None, acentric_factor: Optional[float] = None
    ):
        """
        临界参数如果没有给定, 通过Molecule.tmo进行读取:
            mol.thermo_init()  # some kwargs could pass into, see documentation
            print(mol.thermo.Tc)  # the critical temperature
            print(mol.thermo.Psat)  # the saturation vapor pressure

        如果已经给了，就依照给定值。
        如果既没有给定，通过Molecule.tho无法正常读取，则报错！！！
        Args:
            guest:
            critical_temperature:
            critical_pressure:
            acentric_factor:

        Returns:

        Raises:

        """

    def _make_frame_file(self, frame: Molecule):
        """"""

    def _make_input_script(self, *frame_mols: dict[Molecule, Molecule], simulation_type, cycles):
        """"""


        return dedent(f"""
            SimulationType                {simulation_type}
            NumberOfCycles                {cycles}
            NumberOfInitializationCycles  {init_cycles}
            PrintEvery                    {print_every}
            RestartFile                   no
            
            Forcefield                    {forcefield}
            CutOff                        12.8
            ChargeMethod                  Ewald
            EwaldPrecision                1e-6
            UseChargesFromMOLFile         {is_mol}
            
            Framework                     0
            FrameworkName                 streamed
            InputFileType                 cif
            UnitCells                     {a} {b} {c}
            HeliumVoidFraction            {helium_void_fraction}
            ExternalTemperature           {temperature}
            ExternalPressure              {pressure}
            
            Movies                        no
            WriteMoviesEvery              100
            
            Component 0 MoleculeName             {molecule_name}
                      StartingBead             0
                      MoleculeDefinition       TraPPE
                      IdealGasRosenbluthWeight 1.0
                      TranslationProbability   1.0
                      RotationProbability      1.0
                      ReinsertionProbability   1.0
                      SwapProbability          1.0
                      CreateNumberOfMolecules  0
        """)


