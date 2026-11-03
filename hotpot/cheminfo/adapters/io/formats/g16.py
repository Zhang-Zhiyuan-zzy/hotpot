"""Gaussian/cclib decoding into the existing factual record and pure GJF helpers."""

from dataclasses import replace
import io
import re
import warnings
import cython
import numpy as np

from ....structure._fields import ATOM_SOURCE_CODES, FIELD_PRESENCE_BITS


def _extract_force_matrix(lines, atomic_numbers):
    # Define the format of force sheet
    # the Force sheet like this:
    #  -------------------------------------------------------------------
    #  Center     Atomic                   Forces (Hartrees/Bohr)
    #  Number     Number              X              Y              Z
    #  -------------------------------------------------------------------
    #       1        8           0.039901671    0.000402574    0.014942530
    #       2        8           0.017381613    0.001609531    0.006381231
    #       3        6          -0.092853735   -0.025654844   -0.005885898
    #       4        6           0.067801154    0.024130172   -0.022794721
    #       5        8          -0.023702905    0.005486251   -0.004938175
    #       6        8          -0.006359715   -0.008543465    0.010350815
    #       7       55          -0.002168084    0.002569781    0.001944217
    #  -------------------------------------------------------------------
    force_head1 = re.compile(r'\s*Center\s+Atomic\s+Forces\s\(Hartrees/Bohr\)\s*')
    force_head2 = re.compile(r'\s*Number\s+Number\s+X\s+Y\s+Z\s*')
    sheet_line = re.compile(r'\s*----+\s*')

    HEAD_LINES_NUM = 3  # the offset line to write the header

    head_lines = [i for i, line in enumerate(lines) if force_head1.match(line)]

    all_forces = []
    i: cython.int
    for i in head_lines:
        # enhance the inspection of Force sheet head
        assert force_head2.match(lines[i + 1])
        assert sheet_line.match(lines[i + 2])

        rows = 0
        forces = []
        while True:

            if sheet_line.match(lines[i + HEAD_LINES_NUM + rows]):
                if len(forces) == len(atomic_numbers):
                    all_forces.append(forces)
                    break
                else:
                    raise ValueError('the number of force vector do not match the number of atoms')

            ac, an, x, y, z = map(
                lambda v: int(v[1]) if v[0] < 2 else float(v[1]),
                enumerate(lines[i + HEAD_LINES_NUM + rows].split())
            )

            # Enhance the inspection
            assert ac == rows + 1
            if atomic_numbers[rows] != an:
                raise ValueError('the atomic number do not match')

            forces.append([x, y, z])

            rows += 1

    return np.array(all_forces)


def _extract_g16_thermo(lines):
    # Grab thermal energy, delta capacity at volume, delta entropy
    anchor_line = 0
    title_pattern = re.compile(r'\s+E \(Thermal\)\s+CV\s+S')
    for i, line in enumerate(lines):
        if title_pattern.match(line):
            anchor_line = i
            break

    if anchor_line != 0:
        thermal_energy, capacity, _ = map(float, re.split(r'\s+', lines[anchor_line + 2].strip())[1:])
        thermal_energy = 0.043361254529175 * thermal_energy  # kcal to ev
        capacity = 0.043361254529175e-3 * capacity  # cal to ev

        return thermal_energy, capacity

    else:
        return None, None


def _read_cclib(src, src_type, fmt):
    import cclib

    try:
        if src_type == 'str':
            data = cclib.ccopen(io.StringIO(src)).parse()
        elif src_type == 'path':
            data = cclib.ccopen(src).parse()
        elif src_type == 'IOString':
            data = cclib.ccopen(src).parse()
        else:
            raise RuntimeError(f'the source type {type(src)} have not been supported in cclib')

    except (RuntimeError, AttributeError):
        data = None

    if not data:
        raise IOError(f'Cannot read any calculational data from {src}')

    if not hasattr(data, 'atomnos'):
        raise ValueError(f'read a incorrect MOL file!! {fmt}')
    return data


def _cclib_frame_indices(frames, coordinate_count, atom_count):
    """Associate only observations whose supplied scope identifies a frame."""
    names = ("energy", "partial_charge", "gibbs", "force", "zero_point",
             "spin_mult", "thermo", "capacity")
    selectors = {}
    for name, value in zip(names, frames[1:]):
        if value is None:
            rows = (None,) * coordinate_count
        elif np.ndim(value) == 0:
            rows = ("whole",) * coordinate_count
        elif name == "partial_charge" and np.shape(value) == (atom_count,):
            rows = (None,) * (coordinate_count - 1) + ("whole",)
        elif len(value) == coordinate_count:
            rows = tuple(range(coordinate_count))
        else:
            rows = (None,) * coordinate_count
            if len(value):
                warnings.warn(
                    f"frame_observation_unmapped field={name} "
                    f"coordinates={coordinate_count} observations={len(value)}",
                    RuntimeWarning, stacklevel=3,
                )
        selectors[name] = rows
    return selectors


def _cclib_record(data, src, src_type, initial):
    coordinates = np.array(data.atomcoords, copy=True) if hasattr(data, 'atomcoords') else None
    energies = np.array(data.scfenergies, copy=True) if hasattr(data, 'scfenergies') else None
    partial_charges = (np.array(data.atomcharges["mulliken"], copy=True)
                       if "mulliken" in getattr(data, "atomcharges", {}) else None)
    gibbs = getattr(data, 'freeenergy', None)
    zero_point = getattr(data, 'zpve', None)
    if gibbs is not None:
        gibbs = gibbs * 27.211386245988
    if zero_point is not None:
        zero_point = zero_point * 27.211386245988
    if src_type == 'str':
        lines = src.splitlines()
    else:
        with open(src) as file:
            lines = file.readlines()
    thermo, capacity = _extract_g16_thermo(lines)
    forces = _extract_force_matrix(lines, data.atomnos)
    frames = (coordinates, energies, partial_charges, gibbs, forces,
              zero_point, data.mult, thermo, capacity)
    from ....structure.conformer import _frame_records_from_values

    frame_indices = _cclib_frame_indices(frames, len(coordinates), len(data.atomnos))
    source_frames, conformer_results, frame_selection = _frame_records_from_values(
        tuple(range(len(data.atomnos))), frames, -1,
        producer="cclib", method="Gaussian/cclib",
        units=(("coordinates", "angstrom"), ("energy", "eV"),
               ("partial_charge", "e"), ("gibbs", "eV"),
               ("force", "Hartree/Bohr"), ("zero_point", "eV"),
               ("thermo", "eV"), ("capacity", "eV/K")),
        frame_indices=frame_indices,
    )
    values = np.zeros((len(data.atomnos), 19), dtype="<f8")
    values[:, 0] = data.atomnos
    # The same selected last frame formerly loaded by conformer_load(-1).
    values[:, 10:13] = coordinates[-1]
    presence = np.full(len(values), sum(FIELD_PRESENCE_BITS[index] for index in (0, 10, 11, 12)), dtype="<u4")
    sources = np.zeros(values.shape, dtype="u1")
    sources[:, (0, 10, 11, 12)] = ATOM_SOURCE_CODES["cclib"]
    record = replace(
        initial, atom_count=len(values), atom_values=values.tobytes(),
        atom_presence=presence.tobytes(), atom_sources=sources.tobytes(),
        partial_charge_result=None,
        charge=(True, data.charge, "cclib"), spin_mult=(True, data.mult, "cclib"),
        source_frames=source_frames, conformer_results=conformer_results,
        frame_selection=frame_selection,
    )
    return record


def _gjf_link0(link0: str):
    """ TODO: more work need """
    return link0


def _gjf_route(route: str):
    """ TODO: more work need """
    return route


def from_gjf(script: str) -> "Molecule":
    """ Given a gjf string, return a Molecule object """
