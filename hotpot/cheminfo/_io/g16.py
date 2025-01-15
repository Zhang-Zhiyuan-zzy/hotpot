"""
python v3.9.0
@Project: hotpot
@File   : g16
@Auther : Zhiyuan Zhang
@Data   : 2024/12/13
@Time   : 14:23
"""
import os
from ._io import MolWriter


def _gjf_link0(link0: str):
    """ TODO: more work need """
    return link0

def _gjf_route(route: str):
    """ TODO: more work need """
    return route

@MolWriter.add_plugin('gjf', 'pre')
def to_gjf_preprocess(writer, mol, *args, **kwargs):

    if not kwargs.get('miss_charge_calc', False) or not writer.kwargs.get('miss_charge_calc', False):
        mol.charge = mol.calc_mol_default_charge()

    writer.ob_opt.update({'b': None})
    print(f"mol charge: {mol.charge}")

@MolWriter.add_plugin('gjf', 'post')
def to_gjf(writer, script, mol, *args, **kwargs) -> str:
    """ Given a molecule, return a Gaussian16 gjf string """
    lines = script.splitlines()
    lines[0] = _gjf_link0(kwargs.get('link0') or f'%nproc={min(16, os.cpu_count() // 2)}\n%mem=8GB')
    lines[1] = _gjf_route(kwargs.get('route') or '#p opt/freq b3lyp/3-21g')
    lines[3] = f"Hotpot Molecule: {mol.formula}"
    lines[5] = f"{mol.charge}  {mol.spin_mult or mol.default_spin_mult}"

    script = "\n".join(lines) + '\n\n'

    return script

def from_gjf(script: str) -> "Molecule":
    """ Given a gjf string, return a Molecule object """



from ..core import Molecule
