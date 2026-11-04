"""Historical contact checks with current rules; patched stage runs no optimizer."""

import json
from pathlib import Path

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.operations.forcefields.acceptance import evaluate_structure_acceptance
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord, StageStatus
from tests.test_pipeline.test_sample_containment import _FFReport, _prepare


@pytest.fixture
def observed_geometry():
    data = json.loads((Path(__file__).parent / 'fixtures/stage12_case21_contact.json').read_text())
    numbers = [row[0] for row in data['atoms']]
    graph_hydrogens = [0] * len(numbers)
    for first, second, _, _ in data['bonds']:
        graph_hydrogens[first] += numbers[second] == 1
        graph_hydrogens[second] += numbers[first] == 1
    mol = Molecule()
    for index, (number, charge, *coordinates) in enumerate(data['atoms']):
        mol.create_atom(atomic_number=number, formal_charge=charge,
                        implicit_hydrogens=graph_hydrogens[index],
                        coordinates=coordinates)
    for first, second, order, kind in data['bonds']:
        mol.add_bond(first, second, order, bond_kind=BondKind(kind))
    return data, mol


def _contact(report):
    return next(check for check in report.checks
                if check.name == 'atom_too_close' and check.atom_indices == (28, 54))


@pytest.mark.parametrize('rigid_transform', [False, True])
def test_standard_acceptance_retains_actual_historical_contact(observed_geometry, rigid_transform):
    data, mol = observed_geometry
    original = np.asarray([row[2:] for row in data['atoms']], dtype=float)
    assert data['provenance']['case_id'] == '0021:34111955a32a54f1'
    assert data['provenance']['root_seed'] == 44
    assert len(mol.atoms) == 61
    assert len(mol.bonds) == 72
    assert tuple(atom.atomic_number for atom in mol.atoms) == tuple(row[0] for row in data['atoms'])
    assert tuple(atom.formal_charge for atom in mol.atoms) == tuple(row[1] for row in data['atoms'])
    assert np.array_equal(mol.coordinates, original)
    assert {(bond.a1idx, bond.a2idx, bond.bond_order, bond.bond_kind.value)
            for bond in mol.bonds} == {tuple(row) for row in data['bonds']}
    assert not any(atom is mol.atoms[54] for atom in mol.atoms[28].neighbours)
    if rigid_transform:
        angle = 0.37
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.)))
        mol.coordinates = original @ rotation.T + np.array((3.25, -2.5, 1.75))
    before = mol.coordinates.copy()
    report = evaluate_structure_acceptance(mol, level='standard')
    finding = _contact(report)
    assert report.level == 'standard'
    assert report.passed is False
    assert finding.passed is False
    assert finding.severity == 'error'
    assert finding.bond_indices == ()
    assert finding.measured == pytest.approx(1.0190689996600342, abs=1e-12)
    assert finding.threshold == pytest.approx(1.408, abs=1e-12)
    assert finding.measured == pytest.approx(data['observed_contact']['measured'], abs=1e-12)
    assert finding.threshold == pytest.approx(data['observed_contact']['threshold'], abs=1e-12)
    assert np.isfinite(mol.coordinates).all()
    assert np.array_equal(mol.coordinates, before)


@pytest.mark.parametrize('sample_break', [False, True])
def test_actual_contact_stage_control_preserves_denominator_and_trailing_sample(
        observed_geometry, sample_break, monkeypatch, tmp_path):
    from hotpot.cheminfo.operations.forcefields import stage

    _, contact_mol = observed_geometry
    contact_report = evaluate_structure_acceptance(contact_mol, level='standard')
    argv = ('--sample-break-on', 'FF:atom_too_close') if sample_break else ()
    prepared, payload, context, calls = _prepare('ff', monkeypatch, tmp_path, argv=argv)
    records = (payload.records[0], MolecularRecord(contact_mol), payload.records[2])
    payload = MolecularPayload(records)
    before = contact_mol.coordinates.copy()

    def observed_report_only(molecule, options, **kwargs):
        # Reuse the actual evaluated geometry; this patch performs no optimization.
        calls.append(molecule)
        report = contact_report if molecule is contact_mol else evaluate_structure_acceptance(molecule, level='standard')
        return _FFReport(report)

    monkeypatch.setattr(stage, 'execute_forcefield_route', observed_report_only)
    result = prepared.execute(payload, context)
    rows = result.report['sample_outcomes']
    assert calls == [record.molecule for record in records]
    assert result.status is StageStatus.SUCCEEDED
    assert result.break_pipeline is False
    assert tuple(result.report[key] for key in (
        'input_count', 'attempted_count', 'output_count', 'unavailable_count',
        'sample_break_count', 'not_attempted_count',
    )) == (3, 3, 2 if sample_break else 3, 0, int(sample_break), 0)
    assert [row['input_index'] for row in rows] == [0, 1, 2]
    assert [row['report_index'] for row in rows] == [0, 1, 2]
    finding = next(finding for finding in rows[1]['findings']
                   if finding['code'] == 'FF:atom_too_close' and finding['atom_indices'] == (28, 54))
    actual = _contact(contact_report)
    assert finding['measured'] == actual.measured
    assert finding['threshold'] == actual.threshold
    assert finding['severity'] == actual.severity
    assert finding['passed'] is False
    assert np.array_equal(contact_mol.coordinates, before)
    if sample_break:
        assert [row['outcome'] for row in rows] == ['returned', 'sample-break', 'returned']
        assert [row['output_index'] for row in rows] == [0, None, 1]
        assert result.payload.records == (records[0], records[2])
        assert rows[1]['observation_path'] == 'observations/0001.sdf'
        observation = context.stage_directory / rows[1]['observation_path']
        assert observation.is_file()
        assert '$$$$' in observation.read_text()
        assert any(str(artifact.relative_path).endswith('observations/0001.sdf') for artifact in result.artifacts)
    else:
        assert [row['outcome'] for row in rows] == ['returned'] * 3
        assert [row['output_index'] for row in rows] == [0, 1, 2]
        assert result.payload.records == records
