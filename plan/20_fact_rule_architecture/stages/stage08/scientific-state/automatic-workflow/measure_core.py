import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path.cwd()))

import hotpot
from hotpot.cheminfo.operations.calculation.scientific_state import profiles
from hotpot.cheminfo.operations.calculation.scientific_state.facts import (
    extract_molecular_facts, _state_request_from_molecule,
    _source_radical_electrons, _molecule_state_source_identity,
)
from hotpot.cheminfo.operations.calculation.scientific_state.manifest import _coordination_screening_manifest
from hotpot.cheminfo.operations.calculation.scientific_state.operation_requirements import _workflow_requirements
from hotpot.cheminfo.operations.calculation.scientific_state.execution_selection import project_state
from hotpot.cheminfo.operations.calculation.scientific_state.state_resolver import resolve_scientific_state
from hotpot.cheminfo.rules.chemistry.scientific_state.request import StateRequest
from hotpot.cheminfo.structure._commit import _capture_source, _commit_molecular_declarations
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.plugins._harness.contracts import NativeProcessResult
from hotpot.plugins.xtb.contracts import XTBBackendInfo, XTBMethod
from hotpot.plugins.xtb.state_projection import XTBBackendContext
from tests.test_cheminfo.calculator.test_automatic_scientific_state import _HOLDOUTS, _molecule, _two_s

OUT = Path('/tmp/hotpot_plan20/short/automatic-state-core-20261105')
encoded, payload = profiles._profile_payload()
production_digest = hashlib.sha256(encoded).hexdigest()
payload = {**payload, 'activation_status': 'active'}
candidate_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
profiles._profile_payload = lambda: (candidate_bytes, payload)
profiles._coordination_screening_profile.cache_clear()
backend = XTBBackendContext(XTBBackendInfo(
    Path('/not-executed/xtb'), '6.7.1', None, '0' * 64,
    NativeProcessResult((), Path('/tmp'), 0, 'stubbed; no capability/backend execution', '', 0.0),
    86, 86,
), XTBMethod.GFN2_XTB)
requirements = _workflow_requirements(None, backend.scientific_operation_requirements)


def resolve(mol):
    start = perf_counter_ns()
    facts = extract_molecular_facts(mol)
    captured = perf_counter_ns()
    request = _state_request_from_molecule(mol, facts, StateRequest())
    radicals = _source_radical_electrons(mol)
    source_identity = _molecule_state_source_identity(mol, facts)
    bound = perf_counter_ns()
    manifest = _coordination_screening_manifest(
        source_atom_formal_charges=request.atom_formal_charges,
        source_component_charges=request.component_charges,
        source_radical_electrons=radicals, source_evidence_identity=source_identity,
    )
    manifested = perf_counter_ns()
    bundle = resolve_scientific_state(
        facts, request, policy=manifest.policy, requirements=requirements,
        provider_set=manifest.provider_set, rule_set=manifest.rule_set,
        candidate_budget=manifest.candidate_budget,
    )
    resolved = perf_counter_ns()
    projection = project_state(bundle, backend)
    projected = perf_counter_ns()
    times = dict(zip(('facts_ms', 'source_ms', 'manifest_ms', 'resolve_ms', 'project_ms', 'total_ms'),
                     ((captured-start)/1e6, (bound-captured)/1e6, (manifested-bound)/1e6,
                      (resolved-manifested)/1e6, (projected-resolved)/1e6, (projected-start)/1e6)))
    return bundle, projection, times


def counts(bundle):
    space = bundle.candidate_space
    selected = next((path for path in space.expanded_paths
                     if path.identity == bundle.execution_decision.selected_state_identity), None)
    return {
        'factor_count': len(space.factors),
        'materialized_value_count': sum(len(factor.values) for factor in space.factors),
        'largest_factor': max((len(factor.values) for factor in space.factors), default=0),
        'expanded_paths': len(space.expanded_paths), 'frontier': len(space.frontier),
        'truncated_factor_count': sum(factor.materialization_status.value == 'resource_truncated'
                                      for factor in space.factors),
        'search_status': space.search_status.value,
        'decision': bundle.execution_decision.action.value,
        'charge': None if selected is None else selected.net_charge.value,
        'physical_two_s': None if selected is None else _two_s(bundle, selected),
    }

numbers = {'Eu': 63, 'La': 57, 'Gd': 64, 'Zn': 30, 'Cu': 29}
molecules = tuple(_molecule(case['ligand'], numbers[case['metal']]) for case in _HOLDOUTS)
for _ in range(3):
    for mol in molecules:
        resolve(mol)
all_times = []
case_times = [[] for _ in molecules]
last = []
for repeat in range(10):
    latest = []
    for index, mol in enumerate(molecules):
        bundle, projection, times = resolve(mol)
        all_times.append(times)
        case_times[index].append(times)
        latest.append((bundle, projection))
    last = latest


def summarize(values):
    return {key: {'median': statistics.median(value[key] for value in values),
                  'min': min(value[key] for value in values),
                  'max': max(value[key] for value in values)}
            for key in values[0]}

cases = []
for case, times, (bundle, projection) in zip(_HOLDOUTS, case_times, last):
    record = {'id': case['id'], **counts(bundle), 'timings_ms': summarize(times),
              'projection_status': projection.status.value,
              'backend_uhf': None if projection.projection is None else projection.projection.uhf}
    record['expected_match'] = (record['charge'] == case['expected_charge']
                                and record['physical_two_s'] == case['physical_two_s'])
    cases.append(record)
stress = []
for center_count in (2, 3):
    mol = hotpot.read_mol('.'.join('O' for _ in range(center_count)), 'smi')
    for donor in tuple(mol.atoms):
        center = mol.create_atom(atomic_number=63)
        mol.add_bond(center, donor, bond_kind=BondKind.DATIVE)
    _commit_molecular_declarations(mol, _capture_source(mol),
                                   charge=(False, None, None), spin_mult=(False, None, None))
    bundle, projection, times = resolve(mol)
    stress.append({'eu_centers': center_count, **counts(bundle), 'timings_ms': times,
                   'projection_status': projection.status.value})

result = {
    'date': '2026-11-05',
    'scope': 'Warmed source facts/bind/manifest/resolve/pure projection only; no executable, FF, backend probe, install or scientific energy evaluation.',
    'command': [sys.executable, str(Path(__file__).resolve())],
    'python': sys.version, 'python_executable': sys.executable,
    'platform': platform.platform(), 'hotpot_source': hotpot.__file__,
    'production_profile_sha256': production_digest,
    'isolated_active_profile_sha256': hashlib.sha256(candidate_bytes).hexdigest(),
    'activation_protocol_sha256': hashlib.sha256(Path('plan/20_fact_rule_architecture/stages/stage08/scientific-state/automatic-workflow/activation_protocol.json').read_bytes()).hexdigest(),
    'warmup_passes': 3, 'measured_passes': 10, 'sample_count': len(all_times),
    'candidate_budget': asdict(bundle.candidate_space.candidate_budget),
    'aggregate_ms': summarize(all_times), 'holdouts': cases, 'stress': stress,
    'interpretation': 'Single-process warm-cache wall time; no before/after comparator and no throughput, scalability or speedup claim. Stress samples are single observations.',
}
(OUT / 'measurement.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'sample_count': result['sample_count'], 'aggregate_ms': result['aggregate_ms'],
                  'holdout_matches': sum(case['expected_match'] for case in cases), 'stress': stress}, indent=2))
