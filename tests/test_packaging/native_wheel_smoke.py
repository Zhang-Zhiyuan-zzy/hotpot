"""Check native wheel functionality in separate toolkit load orders."""

import argparse


def hotpot_first() -> None:
    import hotpot
    from hotpot.cheminfo.rules.mathematics.geometry import _geometry_native
    from hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum import GraphSpectrum
    from hotpot.cheminfo.rules.mathematics.graph import _graph_native, relevant_cycles
    from hotpot.cheminfo.adapters.native.substructure import _search_native
    from hotpot.cheminfo.obWrappers import _ob_native
    from hotpot.cheminfo import rdWrappers
    from rdkit import Chem

    assert GraphSpectrum is not None
    assert _geometry_native is not None and _ob_native is not None
    assert _graph_native.cycle_basis is not None
    assert _search_native.NativeProblem is not None
    assert Chem.MolFromSmiles('CCO').GetNumAtoms() == 3
    assert tuple(h.atom_indices for h in hotpot.read_mol('CCO', 'smi').search_substructure('[C]-[O]')) == (frozenset((1, 2)),)
    assert relevant_cycles([(4, 7), (7, 9), (9, 4)]) == ((4, 7, 9),)
    info = rdWrappers.runtime_info()
    assert info.native and info.rdkit_version == '2026.09.1'
    sample = rdWrappers.generate_conformers(hotpot.read_mol('CCO', 'smi'), options=rdWrappers.ETKDGOptions(seed=123))
    assert sample.status is rdWrappers.GenerationStatus.SUCCESS
    assert sample.coordinates.shape == (1, 3, 3)


def rdkit_first() -> None:
    from rdkit import Chem, rdBase
    import importlib.metadata
    import hotpot

    molecule = hotpot.Molecule()
    atom = molecule.create_atom(atomic_number=6)
    probe = Chem.MolFromSmiles('C[C@H](O)F')
    assert atom.atomic_number == 6
    assert Chem.MolToSmiles(probe, isomericSmiles=True) == 'C[C@H](O)F'
    assert rdBase.rdkitVersion == '2026.09.1'
    assert importlib.metadata.version('rdkit') == '2026.9.1'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('hotpot-first', 'rdkit-first'))
    args = parser.parse_args()
    if args.mode == 'hotpot-first':
        hotpot_first()
    else:
        rdkit_first()
    print(f'native wheel smoke passed: {args.mode}')


if __name__ == '__main__':
    main()
