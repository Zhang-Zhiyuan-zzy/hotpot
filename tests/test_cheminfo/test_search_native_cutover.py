from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys

import networkx as nx
import pytest

import hotpot as hp
from hotpot.cheminfo.search.native import _NativeMappingSource


def _reject_graph_matcher(*args, **kwargs):
    raise AssertionError("production search constructed NetworkX GraphMatcher")


def test_every_public_search_method_uses_the_native_mapping_source(monkeypatch):
    monkeypatch.setattr(
        nx.algorithms.isomorphism,
        "GraphMatcher",
        _reject_graph_matcher,
    )
    searcher = hp.Searcher(hp.Substructure.from_smarts("CO"))
    molecule = hp.read_mol("CCO", "smi")

    assert searcher.has_match(molecule)
    assert len(tuple(searcher.iter_mappings(molecule))) == 1
    hits = searcher.search(molecule)
    assert len(hits) == 1
    assert isinstance(hits.graph_matcher, _NativeMappingSource)


def test_direct_recursive_predicate_evaluation_uses_native_search(monkeypatch):
    monkeypatch.setattr(
        nx.algorithms.isomorphism,
        "GraphMatcher",
        _reject_graph_matcher,
    )
    query_atom = hp.Substructure.from_smarts(
        "[#6;$([#6]=[#8])]"
    ).query_atoms[0]
    molecule = hp.read_mol("CC(=O)N", "smi")

    assert [query_atom.match(atom) for atom in molecule.atoms] == [
        False,
        True,
        False,
        False,
    ]


def test_native_has_match_stops_after_the_first_accepted_embedding():
    calls = []

    def accept(atom):
        calls.append(atom.idx)
        return True

    query = hp.Substructure()
    query.add_atom(hp.QueryAtom(any_keyword=accept))

    assert hp.Searcher(query).has_match(hp.read_mol("CCO", "smi"))
    assert calls == [0]


def test_query_and_target_mutation_are_resnapshotted_between_calls():
    query = hp.Substructure()
    query.add_atom(hp.QueryAtom(atomic_number={6}))
    searcher = hp.Searcher(query)
    molecule = hp.read_mol("C", "smi")

    assert searcher.has_match(molecule)
    query.query_atoms[0].kwargs["atomic_number"] = {7}
    assert not searcher.has_match(molecule)

    molecule.atoms[0].atomic_number = 7
    assert searcher.has_match(molecule)


def test_native_mapping_views_remain_read_only():
    mapping = next(
        hp.Searcher(hp.Substructure.from_smarts("C")).iter_mappings(
            hp.read_mol("C", "smi")
        )
    )

    with pytest.raises(TypeError):
        mapping[0] = 1


def test_native_mapping_sequence_is_stable_across_workers():
    searcher = hp.Searcher(hp.Substructure.from_smarts("*.*.*.*"))
    molecule = hp.read_mol("C.C.C.C", "smi")

    def sequence():
        return tuple(
            tuple(mapping[index] for index in range(4))
            for mapping in searcher.iter_mappings(molecule)
        )

    expected = sequence()
    with ThreadPoolExecutor(max_workers=4) as executor:
        observed = tuple(executor.map(lambda _: sequence(), range(8)))

    assert expected
    assert all(result == expected for result in observed)


def test_native_mapping_sequence_is_stable_across_process_hash_seeds():
    program = """
import json
import hotpot as hp

query = hp.Substructure.from_smarts("*.*.*.*")
molecule = hp.read_mol("C.C.C.C", "smi")
sequence = [
    [mapping[index] for index in range(4)]
    for mapping in hp.Searcher(query).iter_mappings(molecule)
]
print(json.dumps(sequence, separators=(",", ":")))
"""
    outputs = []
    for seed in ("1", "7", "41"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", program],
                env=environment,
                text=True,
            ).strip()
        )

    assert outputs[0]
    assert len(set(outputs)) == 1
