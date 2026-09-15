import subprocess
import sys


def test_hotpot_imports_with_rdkit_and_openbabel_in_a_fresh_process():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import hotpot; "
                "from openbabel import openbabel as ob; "
                "from rdkit import Chem; "
                "assert ob.OBReleaseVersion(); "
                "assert Chem.MolFromSmiles('CN') is not None"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
