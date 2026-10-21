from types import SimpleNamespace

import hotpot as hp

from hotpot.cheminfo.AImodels.mca import site_detection
from hotpot.cheminfo.search import native


def _mapping_sequence(source):
    return tuple(
        tuple(sorted(mapping.items()))
        for mapping in source.subgraph_monomorphisms_iter()
    )


def test_ordered_prepared_queries_match_individual_searches_and_share_target(
    monkeypatch,
):
    molecule = hp.read_mol("CC(=O)N", "smi")
    queries = tuple(
        native._prepare_query(hp.Substructure.from_smarts(smarts))
        for smarts in ("C", "C=O", "C", "N", "[#6;$([#6]=[#8])]")
    )
    expected = tuple(
        _mapping_sequence(next(native._iter_mapping_sources((query,), molecule)))
        for query in queries
    )
    pack_calls = 0
    target_calls = 0
    problem_targets = []
    native_module = native._native_module()
    original_pack = native._pack_target_snapshot

    def counted_pack(*args, **kwargs):
        nonlocal pack_calls
        pack_calls += 1
        return original_pack(*args, **kwargs)

    class CountedNativeModule:
        @staticmethod
        def NativeTarget(target_spec):
            nonlocal target_calls
            target_calls += 1
            return native_module.NativeTarget(target_spec)

        @staticmethod
        def NativeProblem(query_spec, target):
            problem_targets.append(target)
            return native_module.NativeProblem(query_spec, target)

    monkeypatch.setattr(native, "_pack_target_snapshot", counted_pack)
    monkeypatch.setattr(native, "_native_module", CountedNativeModule)
    sources = tuple(native._iter_mapping_sources(queries, molecule))

    assert tuple(map(_mapping_sequence, sources)) == expected
    assert pack_calls == 1
    assert target_calls == 1
    assert len({id(target) for target in problem_targets}) == 1


def test_empty_prepared_query_group_does_not_pack_or_create_native_target(
    monkeypatch,
):
    def reject(*args, **kwargs):
        raise AssertionError("empty query group prepared a target")

    monkeypatch.setattr(native, "_pack_target_snapshot", reject)
    monkeypatch.setattr(native, "_native_module", reject)

    assert tuple(
        native._iter_mapping_sources((), hp.read_mol("C", "smi"))
    ) == ()


def test_prepared_query_is_reused_without_recompilation(monkeypatch):
    query = native._prepare_query(hp.Substructure.from_smarts("C=O"))

    def reject_compile(*args, **kwargs):
        raise AssertionError("a prepared query was recompiled")

    monkeypatch.setattr(native, "_compile_query", reject_compile)

    assert _mapping_sequence(
        next(
            native._iter_mapping_sources(
                (query,), hp.read_mol("CC=O", "smi")
            )
        )
    )


def test_prepared_group_unions_all_target_requirements(monkeypatch):
    empty = {
        "full": (),
        "ligand": (),
        "atom_fields": (),
        "bond_aromatic": False,
        "total_h": False,
    }
    queries = (
        native._PreparedQuery(
            {}, True, False, empty | {"full": ("degree",)}
        ),
        native._PreparedQuery(
            {}, False, True, empty | {"ligand": ("ring_sizes",)}
        ),
        native._PreparedQuery(
            {},
            False,
            False,
            empty
            | {
                "full": ("connectivity", "degree"),
                "atom_fields": ("formal_charge",),
                "bond_aromatic": True,
                "total_h": True,
            },
        ),
    )
    captured = {}

    def capture_pack(mol, **requirements):
        captured.update(requirements)
        return {"packed": mol}

    class FakeProblem:
        query_keys = ()

        def __init__(self, query_spec, target):
            pass

        def cursor(self):
            return iter(())

    fake_module = SimpleNamespace(
        NativeTarget=lambda target_spec: target_spec,
        NativeProblem=FakeProblem,
    )
    monkeypatch.setattr(native, "_pack_target_snapshot", capture_pack)
    monkeypatch.setattr(native, "_native_module", lambda: fake_module)

    assert len(tuple(native._iter_mapping_sources(queries, object()))) == 3
    assert captured == {
        "needs_atom_objects": True,
        "needs_bond_objects": True,
        "derived_requirements": {
            "full": ("degree", "connectivity"),
            "ligand": ("ring_sizes",),
            "atom_fields": ("formal_charge",),
            "bond_aromatic": True,
            "total_h": True,
        },
    }


def test_mapped_target_projection_matches_public_mapping_order():
    molecule = hp.read_mol("CC(=O)N", "smi")
    substructure = hp.Substructure.from_smarts("[C:1](=O)N")
    searcher = hp.Searcher(substructure)
    anchor = next(
        atom.idx for atom in substructure.query_atoms if atom.map_number == 1
    )
    source = next(
        native._iter_mapping_sources(
            (native._prepare_query(substructure),), molecule
        )
    )

    assert tuple(source.mapped_target_indices(anchor)) == tuple(
        mapping[anchor] for mapping in searcher.iter_mappings(molecule)
    )


def test_mca_rules_reuse_compiled_queries_and_one_generic_target(monkeypatch):
    pack_calls = 0
    original = native._pack_target_snapshot

    def counted_pack(*args, **kwargs):
        nonlocal pack_calls
        pack_calls += 1
        return original(*args, **kwargs)

    def reject_compile(*args, **kwargs):
        raise AssertionError("MCA recompiled its invariant query rules")

    monkeypatch.setattr(native, "_pack_target_snapshot", counted_pack)
    monkeypatch.setattr(native, "_compile_query", reject_compile)
    sites = site_detection.find_nucleophilic_sites(
        hp.read_mol("CC(=O)N", "smi")
    )

    assert tuple((site.atom_index, site.site_type) for site in sites) == (
        (2, "Amide"),
        (1, "double_bond"),
        (3, "atom_with_lone_pair"),
    )
    assert pack_calls == 1
