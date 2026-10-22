import subprocess
import sys

import pytest


_EXERCISE_BOTH_TOOLKITS = """
hotpot_molecule = hotpot.Molecule()
hotpot_atom = hotpot_molecule.create_atom(atomic_number=6)
assert len(hotpot_molecule.atoms) == 1
assert hotpot_atom.atomic_number == 6

rdkit_molecule = Chem.MolFromSmiles("C[C@H](O)F")
assert rdkit_molecule is not None
assert Chem.MolToSmiles(rdkit_molecule, isomericSmiles=True) == "C[C@H](O)F"
"""


@pytest.mark.parametrize(
    "imports",
    (
        "import hotpot\nfrom rdkit import Chem",
        "from rdkit import Chem\nimport hotpot",
    ),
    ids=("hotpot-then-rdkit", "rdkit-then-hotpot"),
)
def test_hotpot_and_rdkit_supported_import_orders_in_fresh_process(imports):
    completed = subprocess.run(
        (sys.executable, "-c", f"{imports}\n{_EXERCISE_BOTH_TOOLKITS}"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
