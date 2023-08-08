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
            force_field: Union[str, os.PathLike],
            frame: Molecule,
            *guests: Molecule
    ):
        """ Initialization"""
        self.raspa_root = Path(raspa_root)
        self.work_dir = work_dir
        self.force_field = force_field
        self.frame = frame
        self.guests = guests

    def _organize_force_field(self):
        """
        self.force_field可以是路径也可以是力场名称，当输入名称是该名称一般为从data.force_field开始的相对路径。
        无论输入的是路径还是名称，如果对应的文件为RASPA可以直接读取的文件，记录该路径并在值作RASPA运行脚本时将其写入
        如果对应文件为RASPA不能直接处理的文件，则将其中的信息转化为RASPA支持的文件，转化后的文件放到work_dir目录底下，以供调用
        Returns:

        """

    def _make_guest_file(
            self, critical_temperature: Optional[float] = None,
            critical_pressure: Optional[float] = None, acentric_factor: Optional[float] = None
    ):
        """
        临界参数如果没有给定, 通过Molecule.tmo进行读取:
            mol.thermo_init()  # some kwargs could pass into, see documentation
            print(mol.thermo.Tc)  # the critical temperature
            print(mol.thermo.Psat)  # the saturation vapor pressure

        如果已经给了，就依照给定值。
        如果既没有给定，通过Molecule.tmo无法正常读取，则报错！！！
        Args:
            critical_temperature:
            critical_pressure:
            acentric_factor:

        Returns:

        Raises:

        """

    def _make_frame_file(self):
        """"""

    def _make_input_script(self, *args, **kwargs):
        """
        下文只是一个案例，需要想一下怎样构建输入参数与代码是的能够执行的工作尽可能的灵活
        这个函数的底下的相关参数可以从args, kwargs中索取，也可以自己定义

        Returns:

        """

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

    def run(self, *args, **kwargs):
        """"""
        # TODO: 组织力场文件
        self._organize_force_field()

        # TODO: write the frame and guest to work_dir
        self._make_frame_file()
        self._make_guest_file(**kwargs)

        # TODO: 制作输入脚本
        self._make_input_script(*args, **kwargs)

        # TODO: 值作RUN文件
        ...

        # TODO: 运行raspa 通过subprocess模块 或者 os, system 模块


