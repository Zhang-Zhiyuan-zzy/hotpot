"""
python v3.9.0
@Project: hotpot
@File   : g16
@Auther : Zhiyuan Zhang
@Data   : 2024/12/13
@Time   : 14:23
"""
from ._io import MolWriter


def _gjf_link0(link0: str):
    """ TODO: more work need """
    return link0

def _gjf_route(route: str):
    """ TODO: more work need """
    return route

@MolWriter.add_plugin('gjf', 'post')
def to_gjf(script, mol, *args, **kwargs) -> str:
    """ Given a molecule, return a Gaussian16 gjf string """
    lines = script.splitlines()
    lines[0] = _gjf_link0(kwargs.get('link0', '%nproc=16\n%mem=64GB'))
    lines[1] = _gjf_route(kwargs.get('route', '#p opt/freq b3lyp/3-21g'))
    lines[3] = f"Hotpot Molecule: {mol.formula}"
    lines[5] = f"{mol.charge}  {mol.spin_mult or mol.default_spin_mult}"

    script = "\n".join(lines) + '\n\n'

    return script

def from_gjf(script: str) -> "Molecule":
    """ Given a gjf string, return a Molecule object """




from ..core import Molecule
