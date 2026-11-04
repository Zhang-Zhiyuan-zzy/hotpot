"""Explicit fake-SDK evidence for lazy CCDC package resources, not CCDC chemistry."""

import textwrap

import pytest

from tests.test_cheminfo.test_import_safety import _run_fresh


@pytest.mark.parametrize("first_attribute", ("solvents", "solvents_smiles"))
def test_ccdc_solvents_load_packaged_members_once_on_first_attribute(first_attribute, tmp_path):
    observation = _run_fresh(
        f"first_attribute = {first_attribute!r}\n" + textwrap.dedent('''
        from collections import OrderedDict
        from contextlib import contextmanager
        import importlib
        from importlib import resources
        from importlib.abc import Loader, MetaPathFinder
        from importlib.machinery import ModuleSpec
        import json
        from pathlib import Path
        import sys
        from types import SimpleNamespace

        import hotpot
        assert not any(name == "ccdc" or name.startswith("ccdc.") for name in sys.modules)
        imports, reads, materialized, resource_owners = [], [], [], []
        active_paths = set()
        original_files, original_as_file = resources.files, resources.as_file

        def tracked_files(owner):
            resource_owners.append(owner)
            return original_files(owner)

        @contextmanager
        def tracked_as_file(resource):
            materialized.append(resource.name)
            with original_as_file(resource) as path:
                active_paths.add(str(path))
                try:
                    yield path
                finally:
                    active_paths.remove(str(path))

        class FakeReader:
            def __init__(self, path):
                assert isinstance(path, str) and path in active_paths and Path(path).is_file()
                reads.append(path)
                self.molecule = SimpleNamespace(smiles="fake:" + Path(path).name)
            def __enter__(self):
                return self
            def __getitem__(self, index):
                assert index == 0
                return self.molecule
            def __exit__(self, exc_type, exc_value, traceback):
                pass

        class FakeSDK(MetaPathFinder, Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in ("ccdc", "ccdc.io"):
                    return ModuleSpec(fullname, self, is_package=fullname == "ccdc")
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                imports.append(module.__name__)
                if module.__name__ == "ccdc.io":
                    module.MoleculeReader = FakeReader

        sys.meta_path.insert(0, FakeSDK())
        resources.files, resources.as_file = tracked_files, tracked_as_file
        module = importlib.import_module("hotpot.plugins.ccdc_api.solvents")
        assert imports == reads == materialized == resource_owners == []
        assert "ccdc" not in sys.modules and "ccdc.io" not in sys.modules
        declared = {resource.name for resource in original_files("hotpot.dataset").joinpath("resources", "molecules", "solvent").iterdir()}
        assert {"methanol.mol2", "acetone.mol2", "benzene.mol2"} <= declared
        first = getattr(module, first_attribute)
        solvents, smiles = module.solvents, module.solvents_smiles
        assert isinstance(solvents, OrderedDict) and isinstance(smiles, set)
        assert first is (solvents if first_attribute == "solvents" else smiles)
        assert set(solvents) == declared
        assert smiles == {"fake:" + name for name in declared}
        assert imports == ["ccdc", "ccdc.io"]
        assert resource_owners == ["hotpot.dataset"]
        assert set(materialized) == declared and len(materialized) == len(declared)
        assert {Path(path).name for path in reads} == declared and len(reads) == len(declared)
        assert not active_paths
        assert module.solvents is solvents and module.solvents_smiles is smiles
        assert len(reads) == len(declared) and len(materialized) == len(declared)
        print(json.dumps({"sdk": "explicit fake", "first_attribute": first_attribute,
                          "imports": imports, "packaged_members": sorted(declared),
                          "reader_paths": reads, "resource_owners": resource_owners,
                          "read_count": len(reads), "as_file_count": len(materialized),
                          "shared_cached_values": True}))
        '''),
        tmp_path,
    )
    assert observation["sdk"] == "explicit fake"
    assert observation["read_count"] == len(observation["packaged_members"])


def test_missing_ccdc_sdk_is_exposed_at_first_attribute_without_empty_fallback(tmp_path):
    observation = _run_fresh('''
        import importlib
        from importlib.abc import MetaPathFinder
        import json
        import sys

        attempts = []
        class MissingSDK(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "ccdc" or fullname.startswith("ccdc."):
                    attempts.append(fullname)
                    raise ModuleNotFoundError("explicit missing CCDC SDK", name="ccdc")
        sys.meta_path.insert(0, MissingSDK())
        module = importlib.import_module("hotpot.plugins.ccdc_api.solvents")
        assert attempts == []
        try:
            module.solvents
        except ModuleNotFoundError as error:
            assert error.name == "ccdc" and str(error) == "explicit missing CCDC SDK"
        else:
            raise AssertionError("missing SDK returned a solvent value")
        assert attempts == ["ccdc"]
        assert "solvents" not in vars(module) and "solvents_smiles" not in vars(module)
        print(json.dumps({"sdk": "explicit missing fake", "attempts": attempts,
                          "failure": "ModuleNotFoundError", "empty_fallback": False}))
        ''', tmp_path)
    assert observation["empty_fallback"] is False
