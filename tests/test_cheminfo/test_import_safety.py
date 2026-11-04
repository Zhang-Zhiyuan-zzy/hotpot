"""Fresh-process facade, actual toolkit ordering and native coexistence contracts."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


_ROOT_EXPORTS = (
    "Molecule", "Atom", "Bond", "BondKind", "Crystal", "MolReader", "MolWriter",
    "read_mol", "MolBundle", "to_pyg_dataset", "is_molecule_input", "to_hotpot_mol",
    "Fragment", "EdgeShoulder", "AtomLink", "AlkylGraft", "BondAdding",
    "AtomReplace", "RingWedge", "AssembleFactory", "Searcher", "Substructure",
    "Hit", "Hits", "Query", "QueryAtom", "QueryBond", "SmartsSemantics",
    "SmartsSyntaxError", "UnsupportedSmartsError", "version",
)


def _run_fresh(source, tmp_path):
    """Keep the child free of pytest/conftest preloads and retain its observation."""
    argv = [sys.executable, "-s", "-B", "-c", textwrap.dedent(source)]
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    record = {
        "argv": argv,
        "cwd": str(Path.cwd()),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    (tmp_path / "fresh-process.json").write_text(json.dumps(record, indent=2) + "\n")
    assert result.returncode == 0, record
    observation = json.loads(result.stdout.splitlines()[-1])
    print(json.dumps(observation, sort_keys=True))
    return observation


def test_root_facade_is_explicit_lazy_and_keeps_public_identity(tmp_path):
    observation = _run_fresh(
        f"expected = {_ROOT_EXPORTS!r}\n" + textwrap.dedent('''
        import importlib
        import json
        from pathlib import Path
        import sys

        toolkits = ("rdkit", "openbabel", "torch", "onnxruntime")
        def loaded():
            return [name for name in sys.modules
                    if any(name == prefix or name.startswith(prefix + ".") for prefix in toolkits)]

        assert loaded() == []
        before = list(sys.path)
        import hotpot
        from hotpot import cheminfo
        declared_version = importlib.import_module("hotpot.__version__").__version__

        assert hotpot.__all__ == expected
        assert cheminfo.__all__ == expected[:-1]
        assert all(getattr(hotpot, name) is getattr(cheminfo, name) for name in expected[:-1])
        assert hotpot.version() == hotpot.__version__ == declared_version
        namespace = {}
        exec("from hotpot import *", namespace)
        assert set(namespace) - {"__builtins__"} == set(expected)
        assert all(namespace[name] is getattr(hotpot, name) for name in expected)
        assert not {"os", "sys", "join", "dirname", "package_root", "tempdir", "_rdkit_chem"}.intersection(vars(hotpot))
        assert sys.path == before
        assert loaded() == []
        product = Path(hotpot.__file__).resolve().parent
        origins = {name: str(Path(module.__file__).resolve())
                   for name, module in sys.modules.items()
                   if (name == "hotpot" or name.startswith("hotpot.")) and getattr(module, "__file__", None)}
        assert all(Path(path).is_relative_to(product) for path in origins.values())
        print(json.dumps({"version": hotpot.version(), "exports": list(hotpot.__all__),
                          "toolkits_loaded": loaded(), "sys_path_unchanged": sys.path == before,
                          "hotpot_origins": origins}))
        '''),
        tmp_path,
    )
    assert observation["exports"] == list(_ROOT_EXPORTS)
    assert observation["toolkits_loaded"] == []
    assert observation["sys_path_unchanged"]


@pytest.mark.parametrize("first_toolkit", ("openbabel", "rdkit"))
def test_actual_hotpot_toolkit_orders_preserve_ethanol_and_native_coexistence(first_toolkit, tmp_path):
    observation = _run_fresh(
        f"first_toolkit = {first_toolkit!r}\n" + textwrap.dedent('''
        import json
        from pathlib import Path
        import sys

        assert not any(name.split(".")[0] in {"rdkit", "openbabel", "torch", "onnxruntime"}
                       for name in sys.modules)
        import hotpot
        assert "rdkit" not in sys.modules and "openbabel" not in sys.modules
        molecule = hotpot.Molecule()
        for number, hydrogens, x in ((6, 3, 0.0), (6, 2, 1.5), (8, 1, 2.9)):
            molecule.create_atom(atomic_number=number, implicit_hydrogens=hydrogens,
                                 formal_charge=0, coordinates=(x, 0.0, 0.0))
        molecule.add_bond(0, 1, 1)
        molecule.add_bond(1, 2, 1)
        molecule.charge = 0
        assert "rdkit" not in sys.modules and "openbabel" not in sys.modules
        if first_toolkit == "openbabel":
            obmol = molecule.to_obmol()
            rdmol = molecule.to_rdmol()
        else:
            rdmol = molecule.to_rdmol()
            assert "openbabel" not in sys.modules
            obmol = molecule.to_obmol()

        from openbabel import openbabel as ob, pybel
        from rdkit import Chem
        import rdkit
        from hotpot.cheminfo.operations.conversion import from_rdmol, obmol2mol
        from hotpot.cheminfo.obWrappers import native as ob_native
        from hotpot.cheminfo.rdWrappers import native as rd_native

        rd_graph = {
            "atomic_numbers": [atom.GetAtomicNum() for atom in rdmol.GetAtoms()],
            "formal_charges": [atom.GetFormalCharge() for atom in rdmol.GetAtoms()],
            "bonds": sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
                            for bond in rdmol.GetBonds()),
            "charge": Chem.GetFormalCharge(rdmol),
        }
        ob_graph = {
            "atomic_numbers": [atom.GetAtomicNum() for atom in ob.OBMolAtomIter(obmol)],
            "formal_charges": [atom.GetFormalCharge() for atom in ob.OBMolAtomIter(obmol)],
            "bonds": sorted((bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, float(bond.GetBondOrder()))
                            for bond in ob.OBMolBondIter(obmol)),
            "charge": obmol.GetTotalCharge(),
        }
        expected = {"atomic_numbers": [6, 6, 8], "formal_charges": [0, 0, 0],
                    "bonds": [(0, 1, 1.0), (1, 2, 1.0)], "charge": 0}
        assert rd_graph == expected and ob_graph == expected
        for restored in (from_rdmol(rdmol, hotpot.Molecule()), obmol2mol(obmol, hotpot.Molecule())):
            assert [atom.atomic_number for atom in restored.atoms] == [6, 6, 8]
            assert [atom.formal_charge for atom in restored.atoms] == [0, 0, 0]
            assert [(bond.atom1.idx, bond.atom2.idx, bond.bond_order) for bond in restored.bonds] == [(0, 1, 1.0), (1, 2, 1.0)]
            assert restored.charge == 0
        if first_toolkit == "openbabel":
            ob_backend = ob_native._native_module()
            rd_backend = rd_native._backend()
        else:
            rd_backend = rd_native._backend()
            ob_backend = ob_native._native_module()
        data = ob_native._native_molecule_data(molecule)
        assert data.atom_count == 3 and data.bond_count == 2
        info = rd_native.runtime_info()
        assert info.native and info.hotpot_version == hotpot.version()
        assert info.rdkit_version == rdkit.__version__
        assert rd_backend.C_API_VERSION == 1
        product = Path(hotpot.__file__).resolve().parent
        native_origins = {backend.__name__: str(Path(backend.__file__).resolve())
                          for backend in (ob_backend, rd_backend)}
        assert all(Path(path).is_relative_to(product) and Path(path).suffix == ".so"
                   for path in native_origins.values())
        assert all(Path(module.__file__).resolve().is_relative_to(product)
                   for name, module in sys.modules.items()
                   if (name == "hotpot" or name.startswith("hotpot.")) and getattr(module, "__file__", None))
        print(json.dumps({"first_toolkit": first_toolkit, "rdkit_graph": rd_graph,
                          "openbabel_graph": ob_graph, "rdkit_version": rdkit.__version__,
                          "openbabel_version": ob.OBReleaseVersion(), "native_origins": native_origins,
                          "native_openbabel_atom_count": data.atom_count,
                          "native_openbabel_bond_count": data.bond_count,
                          "native_rdkit_version": info.rdkit_version}))
        '''),
        tmp_path,
    )
    assert observation["first_toolkit"] == first_toolkit
    assert observation["rdkit_graph"] == observation["openbabel_graph"]


def test_hotpot_imports_with_rdkit_and_openbabel_in_a_fresh_process(tmp_path):
    """Retain the original coexistence observation after actual lazy adapter use."""
    observation = _run_fresh('''
        import json
        import hotpot
        molecule = hotpot.read_mol("CN")
        from openbabel import openbabel as ob
        from rdkit import Chem
        assert ob.OBReleaseVersion()
        assert Chem.MolFromSmiles("CN") is not None
        assert [atom.atomic_number for atom in molecule.atoms] == [6, 7]
        assert molecule.charge == 0
        print(json.dumps({"openbabel_version": ob.OBReleaseVersion(),
                          "coimported_smiles": "CN", "charge": molecule.charge}))
        ''', tmp_path)
    assert observation["coimported_smiles"] == "CN"


def test_works_conversion_cold_import_orders_toolkits_and_preserves_cn(tmp_path):
    observation = _run_fresh(
        f"work_dir = {str(tmp_path)!r}\n" + textwrap.dedent('''
        from importlib.abc import MetaPathFinder
        import json
        from pathlib import Path
        import sys

        assert not any(name.split(".")[0] in {"rdkit", "openbabel"}
                       for name in sys.modules)
        import_order = []

        class ImportObserver(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in ("rdkit.Chem", "openbabel"):
                    import_order.append(fullname)
                return None

        sys.meta_path.insert(0, ImportObserver())
        from hotpot.works import convert
        assert import_order == ["rdkit.Chem", "openbabel"]
        from openbabel import pybel
        from rdkit import Chem
        assert convert.pb is pybel

        sdf_dir = Path(work_dir) / "sdf"
        smiles_dir = Path(work_dir) / "smiles"
        sdf_dir.mkdir()
        smiles_dir.mkdir()
        pybel.readstring("smi", "CN").write("sdf", str(sdf_dir / "cn.sdf"))
        convert.convert_sdf_to_smiles(sdf_dir, smiles_dir)
        lines = (smiles_dir / "smi_0.csv").read_text().splitlines()
        assert len(lines) == 1
        smiles = lines[0].removesuffix(",").split()[0]
        molecule = Chem.MolFromSmiles(smiles)
        graph = {
            "atomic_numbers": [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            "bonds": [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
                      for bond in molecule.GetBonds()],
            "charge": Chem.GetFormalCharge(molecule),
        }
        assert graph == {"atomic_numbers": [6, 7], "bonds": [(0, 1, 1.0)], "charge": 0}
        assert len(pybel.readstring("smi", smiles).atoms) == 2
        print(json.dumps({"import_order": import_order, "smiles": smiles, "graph": graph,
                          "pybel_module": convert.pb.__name__, "rdkit_module": Chem.__name__}))
        '''),
        tmp_path,
    )
    assert observation["import_order"] == ["rdkit.Chem", "openbabel"]
    assert observation["graph"] == {"atomic_numbers": [6, 7], "bonds": [[0, 1, 1.0]], "charge": 0}
