import numpy as np
from rdkit import Chem

from mca.conformer import ensure_3d_conformer


def test_degenerate_hotpot_style_conformer_is_reembedded():
    molecule = Chem.MolFromSmiles("CCN")
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    molecule.AddConformer(conformer)

    embedded = ensure_3d_conformer(molecule, seed=42)
    positions = embedded.GetConformer().GetPositions()

    assert embedded.GetConformer().Is3D()
    assert np.max(np.linalg.norm(positions - positions[0], axis=1)) > 1e-6
