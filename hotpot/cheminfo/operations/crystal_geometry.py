"""Read-only periodic geometry observations over captured chemical facts."""

import operator
import warnings

import numpy as np

from .crystal_generation import _arrays, _capture_input
from .crystal_matching import _positive_length
from ..rules.mathematics.geometry.linking import _polygon_linking
from ..structure._crystal import _image
from ..structure._fields import FIELD_INDEX


_XYZ = tuple(FIELD_INDEX[name] for name in ('x', 'y', 'z'))


def _cycle_input(record, cycle):
    cycle = tuple((operator.index(row), _image(image)) for row, image in cycle)
    if any(not 0 <= row < record.atom_count for row, _ in cycle):
        raise ValueError('CRYSTAL-IMAGE-MAP-INTEGRITY: cycle Atom row outside current membership')
    return cycle


def _directed_images(record):
    result = {}
    for (first, second), image in zip(record.bond_endpoints, record.cell[4]):
        result[first, second] = image
        result[second, first] = None if image is None else tuple(-value for value in image)
    return result


def _cycle_geometry(record, cycle, directed):
    """Check real edges and a zero-voltage lift; never invent a closing edge."""
    if len(cycle) < 3:
        return None, (('reason', 'fewer_than_three_vertices'),)
    for index, ((first, start), (second, end)) in enumerate(zip(cycle, cycle[1:] + cycle[:1])):
        edge = first, second
        subjects = (('edge', (cycle[index], cycle[(index + 1) % len(cycle)])),)
        if edge not in directed:
            return None, (('reason', 'open_cycle'), *subjects)
        expected = directed[edge]
        if expected is None:
            return None, (('reason', 'unknown_bond_image'), *subjects)
        supplied = tuple(last - initial for initial, last in zip(start, end))
        if supplied != expected:
            reason = 'nonzero_cycle_voltage' if index == len(cycle) - 1 else 'cycle_edge_image_mismatch'
            return None, (('reason', reason), *subjects, ('expected_image', expected), ('supplied_image', supplied))
    coordinates = _arrays(record)[0][:, _XYZ]
    return tuple(tuple(coordinates[row] + np.asarray(image) @ np.asarray(record.cell[1]))
                 for row, image in cycle), ()


def _link_cycles(record, first, second, tolerance):
    directed = _directed_images(record)
    polygons = []
    for index, cycle in enumerate((first, second)):
        coordinates, issue = _cycle_geometry(record, cycle, directed)
        if issue:
            return None, (('curve_index', index), *issue)
        polygons.append(coordinates)
    return _polygon_linking(*polygons, tolerance=tolerance)


def _emit_findings(findings):
    for code, subjects, measurements in findings:
        warnings.warn(f'{code}: subjects={subjects}; measurements={measurements}', RuntimeWarning, stacklevel=4)


def _linking_number(crystal, first_cycle, second_cycle, tolerance):
    tolerance = _positive_length(tolerance, 'tolerance')
    molecule = crystal._molecule
    if molecule is None:
        raise ValueError('CRYSTAL-UNSUPPORTED-PROJECTION: linking requires contained Molecule facts')
    with molecule._mutation_lock:
        record = _capture_input(crystal, molecule, require_group=False)
        first, second = (_cycle_input(record, cycle) for cycle in (first_cycle, second_cycle))
    value, measurements = _link_cycles(record, first, second, tolerance)
    if value is None:
        _emit_findings((('CRYSTAL-LINKING-UNAVAILABLE', (first, second), measurements),))
    elif value:
        _emit_findings((('CRYSTAL-CLOSED-LOOP-LINK', (first, second),
                         (('linking_number', value), *measurements)),))
    return value
