"""Canonical source frames, generated ensembles and their read-only projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator
from threading import RLock
from typing import Union
import warnings

import numpy as np
from numpy.typing import NDArray

from hotpot.utils import types as _types
from ._identity import _capture_source
from ._results import _ResultAttachment

__all__ = ()

_FRAME_NAMES = ("coordinates", "energy", "partial_charge", "gibbs", "force",
                "zero_point", "spin_mult", "thermo", "capacity")
_OBSERVATION_NAMES = _FRAME_NAMES[1:]
_FRAME_DOMAINS = ("frame_representation", "cell", "constraints")
_SELECTED_DOMAINS = ("frame_representation", "geometry", "cell", "constraints")
_PER_ATOM = frozenset(("partial_charge", "force"))


@dataclass(frozen=True)
class ConformerBatch:
    """Independent immutable coordinate sets in stable provider ordinal order."""

    coordinates: NDArray[np.float64]
    ordinals: tuple[int, ...]

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates)
        immutable = np.frombuffer(coordinates.tobytes(), dtype=coordinates.dtype).reshape(
            coordinates.shape
        )
        object.__setattr__(self, "coordinates", immutable)

    def __reduce__(self):
        return type(self), (self.coordinates, self.ordinals)


def _seal_array(value):
    value = np.asarray(value, dtype=np.float64)
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float64).reshape(value.shape)


def _codec():
    # The complete-record builder imports the finite result owners itself.
    from ._builder import _freeze_value, _canonical_value, _thaw_value
    return _freeze_value, _canonical_value, _thaw_value


@dataclass(frozen=True, eq=False)
class EnergyForceResult:
    atom_uids: tuple[int, ...]
    values: tuple
    frame_indices: tuple
    method: str
    method_version: str | None
    parameters: tuple
    units: tuple
    provenance: tuple

    def __post_init__(self):
        _, canonical, _ = _codec()
        uids = tuple(map(operator.index, self.atom_uids))
        values = tuple(map(canonical, self.values))
        indices = tuple(tuple(index if index in (None, "whole") else operator.index(index)
                              for index in row) for row in self.frame_indices)
        if len(set(uids)) != len(uids) or len(values) != 8 or any(len(row) != 8 for row in indices):
            raise ValueError("Frame observations require unique Atom bindings and eight observation slots")
        for row in indices:
            for index, value in zip(row, values):
                if index is None:
                    continue
                if value == ("none",):
                    raise ValueError("An absent frame observation cannot have a row selector")
                if index != "whole":
                    raw = _codec()[2](value)
                    if index < 0 or index >= len(raw):
                        raise ValueError("Frame observation selector is outside its raw observation history")
        for name, value in (("atom_uids", uids), ("values", values), ("frame_indices", indices),
                            ("parameters", canonical(self.parameters)), ("units", canonical(self.units)),
                            ("provenance", canonical(self.provenance))):
            object.__setattr__(self, name, value)

    def __reduce__(self):
        return type(self), (self.atom_uids, self.values, self.frame_indices, self.method,
                            self.method_version, self.parameters, self.units, self.provenance)

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True, eq=False)
class CoordinateFrame:
    atom_uids: tuple[int, ...]
    coordinates: np.ndarray
    observations: EnergyForceResult | None
    observation_index: int | None
    provenance: tuple

    def __post_init__(self):
        uids = tuple(map(operator.index, self.atom_uids))
        coordinates = _seal_array(self.coordinates)
        if len(set(uids)) != len(uids) or coordinates.shape != (len(uids), 3):
            raise ValueError("Source coordinates require unique Atom bindings and shape (N,3)")
        index = None if self.observation_index is None else operator.index(self.observation_index)
        if ((self.observations is None) != (index is None)
                or (index is not None and not 0 <= index < len(self.observations.frame_indices))):
            raise ValueError("Source observations require an existing observation row")
        object.__setattr__(self, "atom_uids", uids)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "observation_index", index)
        object.__setattr__(self, "provenance", _codec()[1](self.provenance))

    def __reduce__(self):
        return type(self), (self.atom_uids, self.coordinates, self.observations,
                            self.observation_index, self.provenance)

    def __deepcopy__(self, memo):
        return self


@dataclass(frozen=True, eq=False)
class ConformerEnsembleResult:
    atom_uids: tuple[int, ...]
    batch: ConformerBatch
    observations: EnergyForceResult | None
    input_source: object

    def __post_init__(self):
        uids = tuple(map(operator.index, self.atom_uids))
        coordinates = _seal_array(self.batch.coordinates)
        ordinals = tuple(map(operator.index, self.batch.ordinals))
        if (len(set(uids)) != len(uids) or coordinates.ndim != 3
                or coordinates.shape[1:] != (len(uids), 3) or len(coordinates) != len(ordinals)):
            raise ValueError("Generated conformers require shape (K,N,3) and one original ordinal per frame")
        if self.observations is not None and len(self.observations.frame_indices) != len(coordinates):
            raise ValueError("Generated observations require one selector row per conformer")
        object.__setattr__(self, "atom_uids", uids)
        object.__setattr__(self, "batch", ConformerBatch(coordinates, ordinals))

    def __reduce__(self):
        return type(self), (self.atom_uids, self.batch, self.observations, self.input_source)

    def __deepcopy__(self, memo):
        return self


def _raw_values(values):
    if isinstance(values, Mapping):
        return tuple(values.get(name) for name in _FRAME_NAMES)
    values = tuple(values)
    if len(values) != 9:
        raise ValueError("Frame import requires all nine conformer domains")
    return values


def _coordinate_sets(coords, atom_count=None):
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim == 2:
        values = values[None, :, :]
    if (values.ndim != 3 or values.shape[-1] != 3
            or (atom_count is not None and values.shape[1] != atom_count)):
        raise ValueError("Conformer coordinates require shape (N,3) or (K,N,3) with complete Atom membership")
    return values


def _infer_indices(values, count, atom_count, *, report):
    columns = []
    for name, value in zip(_OBSERVATION_NAMES, values):
        if value is None:
            selectors = (None,) * count
        elif np.ndim(value) == 0:
            selectors = ("whole",) * count
        elif ((name == "partial_charge" and np.shape(value) == (atom_count,))
              or (name == "force" and np.shape(value) == (atom_count, 3))):
            selectors = (None,) * (count - 1) + (("whole",) if count else ())
        elif len(value) == count:
            selectors = tuple(range(count))
        else:
            selectors = (None,) * count
            if report:
                warnings.warn(
                    f"frame_observation_unmapped field={name} coordinates={count} observations={len(value)}",
                    RuntimeWarning, stacklevel=3,
                )
        columns.append(selectors)
    return tuple(zip(*columns))


def _make_observations(uids, values, count, *, producer, method=None, method_version=None,
                       parameters=(), units=(), frame_indices=None, provenance=(), report=False):
    freeze, _, _ = _codec()
    if frame_indices is None:
        frame_indices = _infer_indices(values, count, len(uids), report=report)
    elif isinstance(frame_indices, Mapping):
        frame_indices = tuple(tuple(frame_indices.get(name, (None,) * count)[row]
                                    for name in _OBSERVATION_NAMES) for row in range(count))
    if len(frame_indices) != count:
        raise ValueError("Frame observations require one selector row per coordinate frame")
    return EnergyForceResult(
        tuple(uids), tuple(map(freeze, values)), frame_indices,
        str(producer) if method is None else method, method_version,
        freeze(parameters), freeze(units), freeze({"fields": {
            name: {"producer": producer, "method": str(producer) if method is None else method,
                   "method_version": method_version, "parameters": parameters, "units": units,
                   "provenance": provenance}
            for name, value in zip(_OBSERVATION_NAMES, values) if value is not None
        }}),
    )


def _observation_value(observations, row, name):
    if observations is None:
        return None
    column = _OBSERVATION_NAMES.index(name)
    index = observations.frame_indices[row][column]
    if index is None:
        return None
    encoded = observations.values[column]
    if index != "whole" and encoded[0] in ("tuple", "list"):
        return _codec()[2](encoded[1][index])
    value = _codec()[2](encoded)
    return value if index == "whole" else value[index]


def _source_preparation(uids, coords, energy):
    coordinates = _coordinate_sets(coords, len(uids))
    values = [None] * 8
    if energy is not None:
        energies = np.asarray(energy, dtype=float).reshape(-1)
        if len(energies) != len(coordinates):
            raise ValueError("The number of supplied energies must match the coordinate frame count")
        values[0] = energies
    observations = _make_observations(uids, values, len(coordinates), producer="explicit", method="supplied")
    provenance = _codec()[0](("producer", "explicit", "coordinates_unit", "angstrom"))
    return tuple(CoordinateFrame(uids, value, observations, row, provenance)
                 for row, value in enumerate(coordinates))


def _bound_conformers(mol):
    view = object.__new__(Conformers)
    view._owner = mol
    return view


class Conformers:
    """Historical coordinate/observation projection, or a standalone source collection."""

    _attrs = _FRAME_NAMES

    def __init__(self):
        self._owner = None
        self._source_frames = ()
        self._standalone_lock = RLock()

    @property
    def _mutation_lock(self):
        return self._standalone_lock if self._owner is None else self._owner._mutation_lock

    def __getstate__(self):
        with self._mutation_lock:
            return (("source", self._source_frames) if self._owner is None
                    else ("bound", self._owner))

    def __setstate__(self, state):
        role, value = state
        self._owner = None if role == "source" else value
        if role == "source":
            self._source_frames = value
            self._standalone_lock = RLock()

    def __len__(self):
        with self._mutation_lock:
            if self._owner is None:
                return len(self._source_frames)
            return len(self._owner._source_frames) + sum(
                len(attachment.payload.batch.coordinates) for attachment in self._owner._conformer_results)

    def __getitem__(self, idx):
        with self._mutation_lock:
            return {name: self.index_attr(name, idx) for name in self._attrs}

    def __iter__(self):
        with self._mutation_lock:
            return iter(tuple(self.index_attr("coordinates", row) for row in range(len(self))))

    def index_attr(self, name, i):
        if name not in self._attrs:
            raise AttributeError(f"The conformers did not store {name} information")
        with self._mutation_lock:
            if isinstance(i, slice):
                rows = tuple(range(len(self)))[i]
                values = tuple(self.index_attr(name, row) for row in rows)
                if not rows:
                    shape = (0, self._atom_count(), 3) if name == "coordinates" else (0,)
                    return np.empty(shape)
                if all(value is None for value in values):
                    shape = ((self._atom_count(),) if name == "partial_charge" else
                             (self._atom_count(), 3) if name == "force" else ())
                    return np.full((len(rows), *shape), np.nan)
                template = next(value for value in values if value is not None)
                return np.asarray(tuple(np.full(np.shape(template), np.nan) if value is None else value
                                        for value in values))
            uids, coordinates, observations, row, _ = self._entry(i)
            if name == "coordinates":
                return self._complete_coordinates(uids, coordinates).copy()
            value = _observation_value(observations, row, name)
            return np.array(value, copy=True) if isinstance(value, np.ndarray) else _decoded_copy(value)

    def _atom_count(self):
        if self._owner is not None:
            return len(self._owner._atoms)
        return len(self._source_frames[0].atom_uids) if self._source_frames else 0

    def _entry(self, index):
        index = operator.index(index)
        count = len(self)
        if index < 0:
            index += count
        if not 0 <= index < count:
            raise IndexError(f"The number of conformers is {count}, but attempt to access the index {index}")
        sources = self._source_frames if self._owner is None else self._owner._source_frames
        if index < len(sources):
            frame = sources[index]
            uids, coordinates, observations = frame.atom_uids, frame.coordinates, frame.observations
            if self._owner is not None and len(uids) > len(self._owner._atoms):
                expected = _uids(self._owner)
                if set(expected).issubset(uids):
                    positions = {uid: row for row, uid in enumerate(uids)}
                    coordinates = coordinates[[positions[uid] for uid in expected]]
                    observations = _remap_observations(observations, {uid: uid for uid in expected}, fragment=True)
                    uids = expected
            return uids, coordinates, observations, frame.observation_index, None
        index -= len(sources)
        for attachment in self._owner._conformer_results:
            payload = attachment.payload
            if index < len(payload.batch.coordinates):
                uids, coordinates, observations = payload.atom_uids, payload.batch.coordinates[index], payload.observations
                if len(uids) > len(self._owner._atoms):
                    expected = _uids(self._owner)
                    if set(expected).issubset(uids):
                        positions = {uid: row for row, uid in enumerate(uids)}
                        coordinates = coordinates[[positions[uid] for uid in expected]]
                        observations = _remap_observations(observations, {uid: uid for uid in expected}, fragment=True)
                        uids = expected
                return uids, coordinates, observations, index, attachment
            index -= len(payload.batch.coordinates)

    def _complete_coordinates(self, uids, coordinates):
        expected = tuple(range(self._atom_count())) if self._owner is None else _uids(self._owner)
        if not set(expected).issubset(uids):
            raise ValueError("frame_atom_mapping_incomplete: source frame lacks complete current Atom membership")
        rows = {uid: row for row, uid in enumerate(uids)}
        return coordinates[[rows[uid] for uid in expected]]

    def add(self, coords, energy=None):
        if self._owner is not None:
            return self._owner.conformer_add(coords, energy)
        coordinates = _coordinate_sets(coords)
        with self._mutation_lock:
            atom_count = self._atom_count() if self._source_frames else coordinates.shape[1]
            prepared = _source_preparation(tuple(range(atom_count)), coordinates, energy)
            self._source_frames += prepared

    def clear(self):
        if self._owner is not None:
            return self._owner.conformer_clear()
        with self._mutation_lock:
            self._source_frames = ()

    def coordinates(self, i):
        return self.index_attr("coordinates", i)

    def energy(self, i):
        return self.index_attr("energy", i)

    def gibbs(self, i):
        return self.index_attr("gibbs", i)

    def partial_charge(self, i):
        return self.index_attr("partial_charge", i)

    def force(self, i):
        return self.index_attr("force", i)


Conformers.add.__annotations__ = {"coords": _types.ArrayLike,
                                 "energy": Union[_types.ArrayLike, float]}


def _decoded_copy(value):
    freeze, _, thaw = _codec()
    return thaw(freeze(value))


def _uids(mol):
    return tuple(atom._uid for atom in mol._atoms)


def _add_source_frames(mol, coords, energy=None):
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]) as transaction:
        prepared = _source_preparation(_uids(mol), mol.coordinates if coords is None else coords, energy)
        count = len(mol._conformers)
        cursor = mol._conformers_index
        transaction.touch_frames()
        if count:
            index = count + cursor if cursor < 0 else cursor
            if index >= len(mol._source_frames):
                index += len(prepared)
            mol._conformers_index = index - count - len(prepared) if cursor < 0 else index
        mol._source_frames += prepared


def _clear_frames(mol):
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]) as transaction:
        transaction.touch_frames()
        transaction.touch_results()
        mol._source_frames = ()
        mol._conformer_results = ()
        mol._frame_result = None
        mol._conformers_index = 0


def _select_observations(mol, observations, row):
    mol._frame_result = (None if observations is None else _ResultAttachment(
        (observations, row), _capture_source(mol, _SELECTED_DOMAINS)))


def _load_frame(mol, index):
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["conformer_load"]) as transaction:
        uids, coordinates, observations, row, attachment = mol._conformers._entry(index)
        coordinates = mol._conformers._complete_coordinates(uids, coordinates)
        if attachment is not None:
            if attachment.source.owner is not mol._owner_token or attachment.invalidated_by:
                raise RuntimeError("conformer_result_stale: generated frame representation, cell or constraints changed")
        transaction.touch_frames()
        transaction.touch_results()
        mol.coordinates = coordinates
        mol._conformers_index = operator.index(index)
        _select_observations(mol, observations, row)


def _frame_records_from_values(atom_uids, values, index, *, producer, method=None,
                               method_version=None, parameters=(), units=(), frame_indices=None):
    """Prepare nine supplied domains without acquiring or changing a live owner.

    Transport is ``(sources, ((ensemble, invalidation_markers), ...),
    (cursor, ((observations, row), invalidation_markers) | None) | None)``. Subjects
    are the supplied UIDs; complete-record callers supply canonical row IDs.
    """
    values = _raw_values(values)
    uids = tuple(atom_uids)
    if values[0] is None:
        return (), (), None
    coordinates = _coordinate_sets(values[0], len(uids))
    observations = _make_observations(
        uids, values[1:], len(coordinates), producer=producer, method=method,
        method_version=method_version, parameters=parameters, units=units,
        frame_indices=frame_indices, report=True,
    )
    provenance = _codec()[0](("producer", producer, "coordinates_unit", "angstrom"))
    sources = tuple(CoordinateFrame(uids, coords, observations, row, provenance)
                    for row, coords in enumerate(coordinates))
    index = operator.index(index)
    if sources and not -len(sources) <= index < len(sources):
        raise IndexError("Imported frame selection is outside the coordinate history")
    row = index % len(sources) if sources else 0
    return sources, (), (index, ((observations, row), ()) if sources else None)


def _import_frame_values(mol, values, index, *, producer, method=None, method_version=None,
                         parameters=(), units=(), frame_indices=None):
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]):
        sources, generated, selection = _frame_records_from_values(
            tuple(range(len(mol._atoms))), values, index, producer=producer, method=method,
            method_version=method_version, parameters=parameters, units=units,
            frame_indices=frame_indices,
        )
        cursor, selected = (0, None) if selection is None else selection
        # Importing coordinate evidence does not publish a different current geometry.
        if sources and not np.array_equal(mol.coordinates, sources[cursor].coordinates):
            selection = cursor, None
        _install_frame_records(mol, sources, generated, selection)


def _select_frame_observations(mol, values, *, producer, method=None, method_version=None,
                               parameters=(), units=(), provenance=()):
    """Publish a complete observation set at the current facts, without source frames."""
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    if isinstance(values, Mapping):
        values = tuple(values.get(name) for name in _OBSERVATION_NAMES)
    else:
        values = tuple(values)
    if len(values) != 8:
        raise ValueError("Current frame publication requires eight observation domains")
    indices = (tuple(None if value is None else "whole" for value in values),)
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]) as transaction:
        observations = _make_observations(
            _uids(mol), values, 1, producer=producer, method=method, method_version=method_version,
            parameters=parameters, units=units, provenance=provenance, frame_indices=indices,
        )
        transaction.touch_results()
        _select_observations(mol, observations, 0)
    return observations


def _current_frame_value(mol, name):
    with mol._mutation_lock:
        attachment = mol._frame_result
        if attachment is None:
            return None
        if attachment.source.owner is not mol._owner_token or attachment.invalidated_by:
            domains = ", ".join(attachment.invalidated_by)
            raise RuntimeError(f"frame_observation_stale: {domains or 'owner'} changed; select or calculate the frame again")
        observations, row = attachment.payload
        value = _observation_value(observations, row, name)
        return np.array(value, copy=True) if isinstance(value, np.ndarray) else _decoded_copy(value)


def _assign_frame_value(mol, name, value):
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    freeze, _, thaw = _codec()
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]) as transaction:
        previous = mol._frame_result
        column = _OBSERVATION_NAMES.index(name)
        if previous is None:
            values = [freeze(None)] * 8
            selectors = [None] * 8
            attribution = {}
        else:
            observations, row = previous.payload
            values = list(observations.values)
            selectors = list(observations.frame_indices[row])
            if previous.invalidated_by:
                spin = selectors[_OBSERVATION_NAMES.index("spin_mult")]
                selectors = [None] * 8
                selectors[_OBSERVATION_NAMES.index("spin_mult")] = spin
            attribution = thaw(observations.provenance)["fields"]
        values[column] = freeze(value)
        selectors[column] = None if value is None else "whole"
        attribution[name] = {"producer": "explicit", "method": "supplied", "method_version": None,
                             "parameters": (), "units": (), "provenance": ()}
        observations = EnergyForceResult(
            _uids(mol), tuple(values), (tuple(selectors),), "supplied", None,
            freeze(()), freeze(()), freeze({"fields": attribution}),
        )
        transaction.touch_results()
        _select_observations(mol, observations, 0)


def _frame_spin(mol):
    attachment = mol._frame_result
    if attachment is None:
        return None
    observations, row = attachment.payload
    return _observation_value(observations, row, "spin_mult")


def _source_frame_inputs(mol):
    return mol._source_frames


def _frame_numeric_inputs(mol):
    selected = mol._frame_result
    return (tuple(frame.observations for frame in mol._source_frames),
            tuple((attachment.payload.observations, attachment.invalidated_by)
                  for attachment in mol._conformer_results),
            None if selected is None else (selected.payload, selected.invalidated_by),
            mol._conformers_index)


def _frame_transaction_state(mol):
    return (mol._conformers, mol._source_frames, mol._conformer_results,
            mol._conformers_index, mol._frame_result)


def _restore_frame_transaction_state(mol, state):
    (mol._conformers, mol._source_frames, mol._conformer_results,
     mol._conformers_index, mol._frame_result) = state


def _publish_conformer_ensemble(mol, batch, values, selected_index, *, input_source,
                                 method, method_version=None, parameters=(), units=(), provenance=()):
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES
    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]) as transaction:
        if values is None:
            raw = (None,) * 8
        elif isinstance(values, Mapping):
            raw = tuple(values.get(name) for name in _OBSERVATION_NAMES)
        else:
            raw = tuple(values)
            if len(raw) == 9:
                raw = raw[1:]
            if len(raw) != 8:
                raise ValueError("Generated frame publication requires eight observation domains")
        observations = _make_observations(
            _uids(mol), raw, len(batch.coordinates), producer=method, method=method,
            method_version=method_version, parameters=parameters, units=units, provenance=provenance,
        )
        payload = ConformerEnsembleResult(_uids(mol), batch, observations, input_source)
        selected_index = operator.index(selected_index)
        if not 0 <= selected_index < len(payload.batch.coordinates):
            raise IndexError("Generated frame selection is outside the conformer batch")
        # The operation checks its captured source before publishing any factual
        # working-copy output. This retained source is historical input provenance.
        transaction.touch_frames()
        transaction.touch_results()
        before_count = len(mol._conformers)
        mol.coordinates = payload.batch.coordinates[selected_index]
        attachment = _ResultAttachment(payload, _capture_source(mol, _FRAME_DOMAINS))
        mol._conformer_results += (attachment,)
        mol._conformers_index = before_count + selected_index
        _select_observations(mol, observations, selected_index)
    return payload


def _remap_observations(observations, mapping, *, fragment=False):
    """Rebind subjects once, subsetting only fields with an explicit Atom scope."""
    if observations is None:
        return None
    freeze, _, thaw = _codec()
    positions = {uid: row for row, uid in enumerate(observations.atom_uids)}
    kept = tuple((positions[uid], output) for uid, output in mapping.items() if uid in positions)
    uids = tuple(uid for _, uid in kept)
    rows = tuple(row for row, _ in kept)
    values = list(observations.values)
    selectors = [list(row) for row in observations.frame_indices]
    provenance = thaw(observations.provenance)
    whole_evidence = []
    subset = rows != tuple(range(len(observations.atom_uids)))
    for column, name in enumerate(_OBSERVATION_NAMES):
        if fragment and name not in _PER_ATOM:
            if values[column] != ("none",):
                whole_evidence.append((name, values[column]))
            values[column] = freeze(None)
            for row in selectors:
                row[column] = None
        elif subset and name in _PER_ATOM and values[column] != ("none",):
            raw = thaw(values[column])
            shape = np.shape(raw)
            width = len(observations.atom_uids)
            if name == "partial_charge" and shape == (width,):
                values[column] = freeze(np.asarray(raw)[list(rows)])
            elif name == "partial_charge" and len(shape) == 2 and shape[1] == width:
                values[column] = freeze(np.asarray(raw)[:, list(rows)])
            elif name == "force" and shape == (width, 3):
                values[column] = freeze(np.asarray(raw)[list(rows)])
            elif name == "force" and len(shape) == 3 and shape[1:] == (width, 3):
                values[column] = freeze(np.asarray(raw)[:, list(rows)])
    if whole_evidence:
        provenance = dict(provenance, whole_system_observations=tuple(whole_evidence))
    return EnergyForceResult(
        uids, tuple(values), tuple(map(tuple, selectors)), observations.method,
        observations.method_version, observations.parameters, observations.units, freeze(provenance),
    )


def _capture_frame_records(mol, rows=None, include_generated=True):
    """Capture canonical row-bound values, never a parallel all-frame matrix.

    Sources are N16 values. Generated transport entries are ``(N17, markers)``;
    selection is ``(cursor, ((N18, row), markers) | None)`` or None. All returned buffers
    are immutable. Repeated references to an observation share one sealed N18.
    """
    with mol._mutation_lock:
        selected_rows = tuple(range(len(mol._atoms))) if rows is None else tuple(rows)
        mapping = {mol._atoms[row]._uid: output for output, row in enumerate(selected_rows)}
        whole = selected_rows == tuple(range(len(mol._atoms)))
        observations = {}

        def remap(payload):
            if payload is None:
                return None
            key = id(payload)
            if key not in observations:
                observations[key] = _remap_observations(
                    payload, mapping, fragment=not whole or any(uid not in mapping for uid in payload.atom_uids))
            return observations[key]

        sources = []
        source_selected = False
        selected = mol._frame_result
        for frame in mol._source_frames:
            kept = tuple((row, mapping[uid]) for row, uid in enumerate(frame.atom_uids) if uid in mapping)
            provenance = frame.provenance
            if len(kept) != len(frame.atom_uids):
                provenance = _codec()[0]({"source": _codec()[2](frame.provenance),
                                          "whole_system_subjects": frame.atom_uids,
                                          "whole_system_coordinates": frame.coordinates})
            source = CoordinateFrame(tuple(uid for _, uid in kept),
                                     frame.coordinates[[row for row, _ in kept]],
                                     remap(frame.observations), frame.observation_index, provenance)
            sources.append(source)
            if selected is not None and selected.payload == (frame.observations, frame.observation_index):
                source_selected = True
        generated = ()
        if include_generated and whole:
            entries = []
            for attachment in mol._conformer_results:
                payload = attachment.payload
                kept = tuple((row, mapping[uid]) for row, uid in enumerate(payload.atom_uids) if uid in mapping)
                marker = attachment.invalidated_by
                observed = remap(payload.observations)
                if len(kept) != len(payload.atom_uids):
                    marker += (() if "frame_representation" in marker else ("frame_representation",))
                    observed = _observation_evidence(observed, "whole_system_coordinates",
                                                     (payload.atom_uids, payload.batch.coordinates))
                    if payload.observations is not None:
                        observations[id(payload.observations)] = observed
                entries.append((ConformerEnsembleResult(
                    tuple(uid for _, uid in kept),
                    ConformerBatch(payload.batch.coordinates[:, [row for row, _ in kept], :], payload.batch.ordinals),
                    observed, payload.input_source,
                ), marker))
            generated = tuple(entries)
        selected_record = None
        if selected is not None and whole and (include_generated or source_selected):
            payload, row = selected.payload
            marker = selected.invalidated_by
            if selected.source.owner is not mol._owner_token:
                raise RuntimeError("Frame record capture requires the current molecular owner")
            selected_record = ((remap(payload), row), marker)
        cursor = mol._conformers_index if whole else 0
        if not include_generated and whole:
            index = len(mol._conformers) + cursor if cursor < 0 else cursor
            if index < len(sources):
                cursor = index - len(sources) if cursor < 0 else index
            else:
                cursor = 0
        return tuple(sources), generated, ((cursor, selected_record) if sources or generated or selected_record is not None else None)


def _install_frame_records(mol, source_frames, conformer_results, frame_selection):
    """Install canonical row-bound owners and rebind applicability exactly once."""
    from ._commit import _current_transaction
    transaction = _current_transaction(mol)
    if transaction is not None:
        transaction.touch_frames()
        transaction.touch_results()
    mapping = dict(enumerate(_uids(mol)))
    observations = {}

    def remap(payload):
        if payload is None:
            return None
        key = id(payload)
        if key not in observations:
            observations[key] = _remap_observations(payload, mapping)
        return observations[key]

    sources = tuple(CoordinateFrame(
        tuple(mapping[uid] for uid in frame.atom_uids), frame.coordinates,
        remap(frame.observations), frame.observation_index, frame.provenance,
    ) for frame in source_frames)
    ensembles = tuple(_ResultAttachment(ConformerEnsembleResult(
        tuple(mapping[uid] for uid in payload.atom_uids), payload.batch,
        remap(payload.observations), payload.input_source,
    ), _capture_source(mol, _FRAME_DOMAINS), tuple(marker)) for payload, marker in conformer_results)
    cursor, selected = (0, None) if frame_selection is None else frame_selection
    selection = None
    if selected is not None:
        (payload, row), marker = selected
        selection = _ResultAttachment((remap(payload), row), _capture_source(mol, _SELECTED_DOMAINS), tuple(marker))
    mol._source_frames = sources
    mol._conformer_results = ensembles
    mol._conformers_index = operator.index(cursor)
    mol._frame_result = selection


def _copy_source_frames(mol, clone, rows=None):
    """Named whole-copy/component policy: source evidence only, no generated runtime."""
    source_frames, _, selection = _capture_frame_records(mol, rows, include_generated=False)
    cursor, selected = (0, None) if selection is None else selection
    if selected is not None:
        if selected[1] or not -len(source_frames) <= cursor < len(source_frames):
            selected = None
        else:
            frame = source_frames[cursor]
            expected = tuple(range(len(clone._atoms)))
            positions = {uid: row for row, uid in enumerate(frame.atom_uids)}
            if (set(frame.atom_uids) != set(expected)
                    or not np.array_equal(clone.coordinates, frame.coordinates[[positions[uid] for uid in expected]])):
                selected = None
    _install_frame_records(clone, source_frames, (), (cursor, selected) if source_frames or selected is not None else None)



def _observation_evidence(observations, name, value):
    """Keep whole-system evidence outside a projected fragment's observation slots."""
    if observations is None:
        return None
    freeze, _, thaw = _codec()
    provenance = dict(thaw(observations.provenance))
    provenance[name] = value
    return EnergyForceResult(
        observations.atom_uids, observations.values, observations.frame_indices,
        observations.method, observations.method_version, observations.parameters,
        observations.units, freeze(provenance),
    )
