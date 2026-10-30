from __future__ import annotations

import base64
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import typing

import numpy as np
from numpy.typing import NDArray
import pytest

from hotpot.cheminfo import stereo


def test_level_one_value_contracts_are_frozen():
    for contract in (
        stereo.StereoCarrier,
        stereo.StereoAnchor,
        stereo.StereoGroupRef,
        stereo.StereoRecord,
        stereo.HydrogenTransportState,
        stereo.AtomTransportRecord,
        stereo.BondTransportRecord,
        stereo.StereoPayload,
    ):
        assert contract.__dataclass_params__.frozen is True


def test_level_two_report_contracts_are_frozen():
    for contract in (
        stereo.StereoValidationIssue,
        stereo.StereoValidationReport,
        stereo.StereoDerivedDescriptor,
        stereo.StereoAssignmentReport,
        stereo.StereoAlignmentReport,
    ):
        assert contract.__dataclass_params__.frozen is True


def test_transport_enums_cover_non_atom_and_coordination_facts():
    assert tuple(stereo.StereoAnchorKind.__members__) == (
        "ATOM",
        "BOND",
        "AXIS",
    )
    assert tuple(stereo.StereoCarrierKind.__members__) == (
        "ATOM",
        "IMPLICIT_H",
        "LONE_PAIR",
        "VACANCY",
    )
    assert tuple(stereo.StereoGroupKind.__members__) == ("ABS", "AND", "OR")
    assert {
        "TETRAHEDRAL",
        "DOUBLE_BOND",
        "ALLENE",
        "ATROPISOMERIC",
        "SQUARE_PLANAR",
        "TRIGONAL_BIPYRAMIDAL",
        "SQUARE_PYRAMIDAL",
        "OCTAHEDRAL",
        "UNKNOWN",
    } == set(stereo.StereoType.__members__)


def test_group_and_hydrogen_contracts_separate_lineage_and_current_state():
    group = stereo.StereoGroupRef(
        scope="source-molecule-a",
        kind=stereo.StereoGroupKind.OR,
        id=1,
    )
    source_hydrogens = stereo.HydrogenTransportState(
        implicit_hydrogens=1,
        bracket_explicit_hydrogens=0,
        no_implicit=False,
    )
    current_hydrogens = stereo.HydrogenTransportState(
        implicit_hydrogens=0,
        bracket_explicit_hydrogens=0,
        no_implicit=True,
    )
    record = stereo.AtomTransportRecord(
        atom=0,
        isotope=None,
        radical_electrons=None,
        current_hydrogens=current_hydrogens,
        source_hydrogens=source_hydrogens,
        source_namespace="rdkit",
    )

    assert group.scope == "source-molecule-a"
    assert group.kind is stereo.StereoGroupKind.OR
    assert record.source_hydrogens.implicit_hydrogens == 1
    assert record.current_hydrogens.no_implicit is True


def test_level_one_function_signatures_are_stable():
    get_payload = inspect.signature(stereo.get_stereo_payload).parameters
    assert tuple(get_payload) == ("mol",)

    set_payload = inspect.signature(stereo.set_stereo_payload).parameters
    assert tuple(set_payload) == ("mol", "payload")


def test_level_two_function_signatures_are_stable():
    validate = inspect.signature(stereo.validate_stereochemistry).parameters
    assert tuple(validate) == ("mol", "coordinates")
    assert validate["coordinates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert validate["coordinates"].default is None

    assign = inspect.signature(stereo.assign_stereochemistry).parameters
    assert tuple(assign) == ("mol", "coordinates")
    assert assign["coordinates"].kind is inspect.Parameter.KEYWORD_ONLY
    assert assign["coordinates"].default is None

    align = inspect.signature(stereo.align_stereochemistry).parameters
    assert tuple(align) == ("reference", "target", "atom_mapping")


# Frozen public roster, field annotations and defaults from the accepted predecessor.
_PUBLIC_EXPORTS = ('AtomTransportRecord',
 'BondTransportRecord',
 'HydrogenTransportState',
 'StereoAlignmentReport',
 'StereoAnchor',
 'StereoAnchorKind',
 'StereoAssignmentReport',
 'StereoCarrier',
 'StereoCarrierKind',
 'StereoDerivedDescriptor',
 'StereoGroupKind',
 'StereoGroupRef',
 'StereoIssueCode',
 'StereoPayload',
 'StereoRecord',
 'StereoType',
 'StereoValidationIssue',
 'StereoValidationReport',
 'align_stereochemistry',
 'assign_stereochemistry',
 'get_stereo_payload',
 'set_stereo_payload',
 'validate_stereochemistry')
_PUBLIC_FIELDS = {'StereoGroupRef': (('scope', 'str', None), ('kind', 'StereoGroupKind', None), ('id', 'int', None)),
 'StereoCarrier': (('kind', 'StereoCarrierKind', None), ('atom', 'int | None', 'None')),
 'StereoAnchor': (('kind', 'StereoAnchorKind', None), ('atom_indices', 'tuple[int, ...]', None)),
 'StereoRecord': (('anchor', 'StereoAnchor', None),
                  ('carriers', 'tuple[StereoCarrier, ...]', None),
                  ('descriptor', 'str', None),
                  ('group', 'StereoGroupRef | None', None),
                  ('source_namespace', 'str', None),
                  ('source_code', 'str', None)),
 'HydrogenTransportState': (('implicit_hydrogens', 'int | None', None),
                            ('bracket_explicit_hydrogens', 'int | None', None),
                            ('no_implicit', 'bool | None', None)),
 'AtomTransportRecord': (('atom', 'int', None),
                         ('isotope', 'int | None', None),
                         ('radical_electrons', 'int | None', None),
                         ('current_hydrogens', 'HydrogenTransportState | None', None),
                         ('source_hydrogens', 'HydrogenTransportState | None', None),
                         ('source_namespace', 'str | None', None)),
 'BondTransportRecord': (('atom_indices', 'tuple[int, int]', None),
                         ('display_direction', 'str | None', None)),
 'StereoPayload': (('schema_version', 'int', None),
                   ('records', 'tuple[StereoRecord, ...]', None),
                   ('atom_facts', 'tuple[AtomTransportRecord, ...]', None),
                   ('bond_facts', 'tuple[BondTransportRecord, ...]', None)),
 'StereoValidationIssue': (('code', 'StereoIssueCode', None),
                           ('message', 'str', None),
                           ('record_index', 'int | None', 'None'),
                           ('atom_indices', 'tuple[int, ...]', '()')),
 'StereoValidationReport': (('is_valid', 'bool', None),
                            ('issues', 'tuple[StereoValidationIssue, ...]', None)),
 'StereoDerivedDescriptor': (('record_index', 'int', None),
                             ('stereo_type', 'StereoType', None),
                             ('label', 'str', None),
                             ('system', 'str', None),
                             ('is_absolute', 'bool', None)),
 'StereoAssignmentReport': (('payload', 'StereoPayload', None),
                            ('unresolved', 'tuple[StereoValidationIssue, ...]', None),
                            ('derived', 'tuple[StereoDerivedDescriptor, ...]', '()')),
 'StereoAlignmentReport': (('is_equivalent', 'bool', None),
                           ('aligned_payload', 'StereoPayload', None),
                           ('issues', 'tuple[StereoValidationIssue, ...]', None))}
_PUBLIC_ENUMS = {'StereoAnchorKind': (('ATOM', 'atom'), ('BOND', 'bond'), ('AXIS', 'axis')),
 'StereoCarrierKind': (('ATOM', 'atom'),
                       ('IMPLICIT_H', 'implicit_h'),
                       ('LONE_PAIR', 'lone_pair'),
                       ('VACANCY', 'vacancy')),
 'StereoGroupKind': (('ABS', 'abs'), ('AND', 'and'), ('OR', 'or')),
 'StereoType': (('TETRAHEDRAL', 'tetrahedral'),
                ('DOUBLE_BOND', 'double_bond'),
                ('ALLENE', 'allene'),
                ('ATROPISOMERIC', 'atropisomeric'),
                ('SQUARE_PLANAR', 'square_planar'),
                ('TRIGONAL_BIPYRAMIDAL', 'trigonal_bipyramidal'),
                ('SQUARE_PYRAMIDAL', 'square_pyramidal'),
                ('OCTAHEDRAL', 'octahedral'),
                ('UNKNOWN', 'unknown')),
 'StereoIssueCode': (('INVALID_ANCHOR', 'invalid_anchor'),
                     ('INVALID_CARRIER', 'invalid_carrier'),
                     ('INVALID_DESCRIPTOR', 'invalid_descriptor'),
                     ('NON_STEREOGENIC', 'non_stereogenic'),
                     ('DEGENERATE_GEOMETRY', 'degenerate_geometry'),
                     ('COORDINATE_CONTRADICTION', 'coordinate_contradiction'),
                     ('INCOMPLETE_MAPPING', 'incomplete_mapping'),
                     ('UNSUPPORTED', 'unsupported'))}


def _contract_samples():
    carrier = stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, 1)
    anchor = stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,))
    group = stereo.StereoGroupRef("baseline", stereo.StereoGroupKind.AND, 7)
    record = stereo.StereoRecord(anchor, (carrier,), "hotpot.stereo/2/TH/even",
                                 group, "baseline", "TH/even")
    hydrogens = stereo.HydrogenTransportState(1, 0, False)
    atom = stereo.AtomTransportRecord(0, 13, 1, hydrogens, hydrogens, "baseline")
    bond = stereo.BondTransportRecord((0, 1), "wedge")
    payload = stereo.StereoPayload(1, (record,), (atom,), (bond,))
    issue = stereo.StereoValidationIssue(stereo.StereoIssueCode.UNSUPPORTED,
                                        "baseline DTO fixture", 0, (0,))
    derived = stereo.StereoDerivedDescriptor(0, stereo.StereoType.TETRAHEDRAL,
                                            "R", "baseline", True)
    return {type(value).__name__: value for value in (
        carrier, anchor, group, record, hydrogens, atom, bond, payload, issue, derived,
        stereo.StereoValidationReport(False, (issue,)),
        stereo.StereoAssignmentReport(payload, (issue,), (derived,)),
        stereo.StereoAlignmentReport(False, payload, (issue,)),
    )}


def test_exact_public_contract_roster_identity_fields_and_enum_order():
    from hotpot.cheminfo.structure import stereo as contracts

    assert stereo.__all__ == _PUBLIC_EXPORTS
    for name in _PUBLIC_EXPORTS[:18]:
        contract = getattr(stereo, name)
        assert contract is getattr(contracts, name)
        assert contract.__module__ == "hotpot.cheminfo.stereo"
        assert contract.__qualname__ == name
        assert pickle.loads(pickle.dumps(contract, protocol=5)) is contract
        if name in _PUBLIC_ENUMS:
            assert tuple((key, item.value) for key, item in contract.__members__.items()) == _PUBLIC_ENUMS[name]
        else:
            actual = tuple((field.name, field.type,
                            None if field.default is MISSING else repr(field.default))
                           for field in fields(contract))
            assert actual == _PUBLIC_FIELDS[name]
            assert all(field.default_factory is MISSING for field in fields(contract))
            assert contract.__dataclass_params__.frozen


def test_exact_public_annotations_signature_and_type_hint_behavior():
    from hotpot.cheminfo import Molecule

    expected = {
        "get_stereo_payload": ({"mol": "Molecule", "return": "StereoPayload"},
                               {"mol": Molecule, "return": stereo.StereoPayload}),
        "set_stereo_payload": ({"mol": "Molecule", "payload": "StereoPayload", "return": "None"},
                               {"mol": Molecule, "payload": stereo.StereoPayload, "return": type(None)}),
        "validate_stereochemistry": (
            {"mol": "Molecule", "coordinates": "NDArray[np.float64] | None", "return": "StereoValidationReport"},
            {"mol": Molecule, "coordinates": NDArray[np.float64] | None, "return": stereo.StereoValidationReport}),
        "assign_stereochemistry": (
            {"mol": "Molecule", "coordinates": "NDArray[np.float64] | None", "return": "StereoAssignmentReport"},
            {"mol": Molecule, "coordinates": NDArray[np.float64] | None, "return": stereo.StereoAssignmentReport}),
        "align_stereochemistry": (
            {"reference": "Molecule", "target": "Molecule", "atom_mapping": "Mapping[int, int]", "return": "StereoAlignmentReport"},
            {"reference": Molecule, "target": Molecule, "atom_mapping": typing.Mapping[int, int], "return": stereo.StereoAlignmentReport}),
    }
    for name, (annotations, resolved) in expected.items():
        function = getattr(stereo, name)
        assert function.__annotations__ == annotations
        signature = inspect.signature(function)
        assert tuple(signature.parameters) == tuple(key for key in annotations if key != "return")
        assert signature.return_annotation == annotations["return"]
        for key, parameter in signature.parameters.items():
            assert parameter.annotation == annotations[key]
            assert parameter.kind is (inspect.Parameter.KEYWORD_ONLY if key == "coordinates"
                                      else inspect.Parameter.POSITIONAL_OR_KEYWORD)
            assert parameter.default is (None if key == "coordinates" else inspect.Parameter.empty)
        assert "Molecule" not in function.__globals__
        assert function.__globals__["Mapping"] is typing.Mapping
        assert function.__globals__["NDArray"] is NDArray
        assert function.__globals__["np"] is np
        with pytest.raises(NameError, match="Molecule"):
            typing.get_type_hints(function)
        assert typing.get_type_hints(function, localns={"Molecule": Molecule}) == resolved


def test_public_value_pickles_keep_canonical_source_tokens():
    from hotpot.cheminfo.structure._identity import _canonical

    samples = _contract_samples()
    for name in _PUBLIC_EXPORTS[:18]:
        contract = getattr(stereo, name)
        values = tuple(contract) if issubclass(contract, Enum) else (samples[name],)
        for value in values:
            restored = pickle.loads(pickle.dumps(value, protocol=5))
            assert restored == value
            assert type(restored) is contract
            canonical = _canonical(value)
            assert canonical[:3] == ["record" if is_dataclass(value) else "enum",
                                     "hotpot.cheminfo.stereo", name]
            assert _canonical(restored) == canonical


@pytest.mark.parametrize("first_import", ("unpickle", "facade", "identity", "core"))
def test_cold_pickles_resolve_the_same_contracts_in_active_install(first_import, tmp_path):
    import hotpot

    samples = _contract_samples()
    values = tuple((name, getattr(stereo, name),
                    tuple(getattr(stereo, name)) if name in _PUBLIC_ENUMS else (samples[name],))
                   for name in _PUBLIC_EXPORTS[:18])
    encoded = base64.b64encode(pickle.dumps(values, protocol=5)).decode("ascii")
    program = """
import base64, importlib, json, pathlib, pickle, sys
order = sys.argv[2]
if order != "unpickle":
    importlib.import_module({"facade": "hotpot.cheminfo.stereo",
                            "identity": "hotpot.cheminfo.structure._identity",
                            "core": "hotpot.cheminfo.core"}[order])
values = pickle.loads(base64.b64decode(sys.argv[1]))
facade = importlib.import_module("hotpot.cheminfo.stereo")
contracts = importlib.import_module("hotpot.cheminfo.structure.stereo")
for name, contract, instances in values:
    assert contract is getattr(facade, name) is getattr(contracts, name)
    assert contract.__module__ == "hotpot.cheminfo.stereo"
    assert all(type(instance) is contract for instance in instances)
print(json.dumps({"origin": str(pathlib.Path(facade.__file__).resolve()),
                  "contracts": len(values)}))
"""
    environment = os.environ.copy()
    package_parent = str(Path(hotpot.__file__).resolve().parent.parent)
    environment["PYTHONPATH"] = os.pathsep.join([package_parent, *(path for path in sys.path if path)])
    script = tmp_path / "cold_pickle_child.py"
    script.write_text(program, encoding="utf-8")
    argv = [sys.executable, str(script), encoded, first_import]
    (tmp_path / "invocation.json").write_text(json.dumps({
        "argv": argv, "cwd": str(tmp_path.resolve()), "env": environment,
        "resolved_executable": str(Path(sys.executable).resolve()),
        "script": str(script), "timeout_seconds": 30,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stdout_path, stderr_path = tmp_path / "stdout.txt", tmp_path / "stderr.txt"
    result_path = tmp_path / "result.json"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            result = subprocess.run(argv, cwd=tmp_path, env=environment, text=True,
                                    stdout=stdout, stderr=stderr, timeout=30)
        except subprocess.TimeoutExpired:
            result_path.write_text(json.dumps({"status": "timeout", "timeout_seconds": 30}) + "\n",
                                   encoding="utf-8")
            raise
    result_path.write_text(json.dumps({"status": "completed", "returncode": result.returncode}) + "\n",
                           encoding="utf-8")
    stdout, stderr = stdout_path.read_text(encoding="utf-8"), stderr_path.read_text(encoding="utf-8")
    assert result.returncode == 0, stdout + stderr
    observed = json.loads(stdout.strip().splitlines()[-1])
    assert observed == {"origin": str(Path(stereo.__file__).resolve()), "contracts": 18}
