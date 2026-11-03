"""Finite selected-result ownership and operation-boundary source bindings."""

from dataclasses import dataclass, replace
import operator
from typing import Generic, TypeVar

from ._identity import _SourceIdentity, _capture_source, _check_source

__all__ = ()

_T = TypeVar("_T")

_RESULT_ATTRIBUTES = ("_mca_result", "_partial_charge_result")
_REPRESENTATION_FIELDS = frozenset((
    "atomic_number", "formal_charge", "implicit_hydrogens", "is_aromatic",
))
_GEOMETRY_FIELDS = frozenset(("x", "y", "z"))
_MCA_OPERATION = "hotpot.cheminfo.calculator.mca(mol, backend=...)"
_CHARGE_OPERATION = "Molecule.assign_partial_charge()"


@dataclass(frozen=True, eq=False)
class _ResultAttachment(Generic[_T]):
    payload: _T
    source: _SourceIdentity
    invalidated_by: tuple[str, ...] = ()


def _capture_mca_source(mol):
    return _capture_source(mol, ("mca_representation", "geometry"))


def _current_mca_result(mol):
    """Project a selected payload under its owner lock without source hashing."""
    attachment = mol._mca_result
    if attachment is None:
        raise AttributeError(f"MCA has not been calculated; call {_MCA_OPERATION} first")
    if attachment.source.owner is not mol._owner_token:
        raise RuntimeError(f"MCA source owner changed; call {_MCA_OPERATION} again")
    if attachment.invalidated_by:
        domains = ", ".join(attachment.invalidated_by)
        raise RuntimeError(f"MCA source dependency {domains} changed; call {_MCA_OPERATION} again")
    return attachment.payload


def _select_mca_result(mol, payload, source, atom_uids):
    """Check one complete prediction and atomically select its exact DTO."""
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES

    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["explicit"]) as transaction:
        _check_source(mol, source)
        atoms = mol._atoms
        if tuple(atom._uid for atom in atoms) != tuple(atom_uids):
            raise RuntimeError("MCA Atom membership changed before result selection")
        predictions = payload.atom_predictions
        if (len(predictions) != len(atoms)
                or any(prediction.atom_index != row or prediction.element != atom.symbol
                       for row, (atom, prediction) in enumerate(zip(atoms, predictions)))):
            raise ValueError("MCA prediction requires complete ordered Atom index and element bindings")
        site_rows = tuple(site.atom_index for site in payload.sites)
        if (len(site_rows) != len(set(site_rows))
                or any(site.atom_index < 0 or site.atom_index >= len(atoms)
                       or site.element != predictions[site.atom_index].element for site in payload.sites)):
            raise ValueError("MCA site prediction requires current Atom index and element bindings")
        transaction.touch_results()
        mol._mca_result = _ResultAttachment(payload, source)
    return payload


def _prepare_result_updates(mol, domains, transaction):
    if "representation" in domains:
        domains = (*domains, "mca_representation", "charge_representation")
    updates = []
    for name in _RESULT_ATTRIBUTES:
        before = getattr(mol, name)
        if before is None:
            continue
        subscribed = tuple(domain for domain, _ in before.source.dependencies)
        markers = tuple(domain for domain in subscribed
                        if domain in domains and domain not in before.invalidated_by)
        if markers:
            after = _ResultAttachment(before.payload, before.source, before.invalidated_by + markers)
            updates.append((name, before, after))
    if updates and transaction is not None:
        transaction.touch_results()
    return tuple(updates)


def _prepare_atom_result_updates(mol, changed_fields, transaction):
    """Prepare finite attachment replacements; native Undo owns publication."""
    changed_fields = frozenset(changed_fields)
    domains = (("representation",) if changed_fields & _REPRESENTATION_FIELDS else ())
    if changed_fields & _GEOMETRY_FIELDS:
        domains += ("geometry",)
    return _prepare_result_updates(mol, domains, transaction)


def _invalidate_results(mol, domains, transaction=None):
    from ._commit import _current_transaction

    with mol._mutation_lock:
        if transaction is None:
            transaction = _current_transaction(mol)
        for name, _, after in _prepare_result_updates(mol, domains, transaction):
            setattr(mol, name, after)


def _copy_mca_result(mol, clone):
    """Only named whole-copy may rebind a selected historical payload."""
    attachment = mol._mca_result
    if attachment is not None:
        if not attachment.invalidated_by:
            _check_source(mol, attachment.source)
        clone._mca_result = _ResultAttachment(
            attachment.payload, _capture_mca_source(clone), attachment.invalidated_by,
        )


@dataclass(frozen=True)
class PartialChargeSet:
    atom_uids: tuple[int, ...]
    values: tuple[float | None, ...]
    method: str
    method_version: str | None
    parameters: tuple[tuple[str, str | None], ...]
    units: str
    provenance: tuple[int, ...]

    def __post_init__(self):
        from ._fields import ATOM_SOURCE_CODES

        uids = tuple(map(operator.index, self.atom_uids))
        values = tuple(None if value is None else float(value) for value in self.values)
        parameters = tuple(tuple(entry) for entry in self.parameters)
        provenance = tuple(map(operator.index, self.provenance))
        if (len(set(uids)) != len(uids) or len(values) != len(uids)
                or len(provenance) != len(uids)):
            raise ValueError("Partial charge values and provenance require complete unique Atom bindings")
        if (type(self.method) is not str or self.units != "e"
                or (self.method_version is not None and type(self.method_version) is not str)
                or any(len(entry) != 2 or type(entry[0]) is not str
                       or (entry[1] is not None and type(entry[1]) is not str) for entry in parameters)
                or any(code not in ATOM_SOURCE_CODES.values() for code in provenance)):
            raise ValueError("Partial charges require a named method, effective parameters, electron units and known producers")
        for name, value in (("atom_uids", uids), ("values", values),
                            ("parameters", parameters), ("provenance", provenance)):
            object.__setattr__(self, name, value)

    def __deepcopy__(self, memo):
        return self


def _capture_partial_charge_source(mol):
    return _capture_source(mol, ("charge_representation", "geometry"))


def _current_partial_charge_attachment(mol):
    """Inspect the finite applicability marker without dependency hashing."""
    attachment = mol._partial_charge_result
    if attachment is None:
        return None
    if attachment.source.owner is not mol._owner_token:
        raise RuntimeError(f"Partial charge source owner changed; call {_CHARGE_OPERATION} again")
    if attachment.invalidated_by:
        domains = ", ".join(attachment.invalidated_by)
        raise RuntimeError(f"Partial charge source dependency {domains} changed; call {_CHARGE_OPERATION} again")
    return attachment


def _current_partial_charge_result(mol):
    """Return a complete current selection for an explicit consumer request."""
    attachment = _current_partial_charge_attachment(mol)
    if attachment is None:
        return None
    if attachment.payload.atom_uids != tuple(atom._uid for atom in mol._atoms):
        raise RuntimeError("Partial charge Atom membership changed; recalculate before projection")
    return attachment.payload


def _partial_charge_projection(mol):
    """Ordered optional observations for numeric views and task encodings."""
    attachment = mol._partial_charge_result
    if attachment is None:
        return (None,) * len(mol._atoms)
    if attachment.source.owner is not mol._owner_token:
        raise RuntimeError(f"Partial charge source owner changed; call {_CHARGE_OPERATION} again")
    if attachment.invalidated_by:
        return (None,) * len(mol._atoms)
    payload = attachment.payload
    if payload.atom_uids != tuple(atom._uid for atom in mol._atoms):
        raise RuntimeError("Partial charge Atom membership changed; recalculate before projection")
    return payload.values


def _atom_partial_charge(atom):
    mol = atom._mol
    with mol._mutation_lock:
        entry = mol._topology_index.get(id(atom))
        if (atom._uid_owner is not mol._owner_token or entry is None
                or mol._atoms[entry[0]] is not atom):
            return None
        attachment = _current_partial_charge_attachment(mol)
        if attachment is None:
            return None
        payload = attachment.payload
        row = entry[0]
        if row >= len(payload.atom_uids) or payload.atom_uids[row] != atom._uid:
            raise RuntimeError("Partial charge Atom membership changed; recalculate before reading")
        return payload.values[row]


def _select_partial_charge_result(mol, payload, source, atom_uids):
    """Select one complete set in the same journal as factual writes."""
    from ._commit import _FactTransaction
    from ._fields import ATOM_SOURCE_CODES

    with _FactTransaction(mol, producer=ATOM_SOURCE_CODES["partial_charge"]) as transaction:
        _check_source(mol, source)
        uids = tuple(atom._uid for atom in mol._atoms)
        if uids != tuple(atom_uids):
            raise RuntimeError("Partial charge Atom membership changed before result selection")
        if payload.atom_uids != uids:
            raise ValueError("Partial charge selection requires complete ordered Atom bindings")
        transaction.touch_results()
        mol._partial_charge_result = _ResultAttachment(payload, source)
    return payload


def _import_partial_charges(mol, values, *, producer, method="supplied",
                            method_version=None, parameters=()):
    """Select an already supplied complete ordered set without calculation."""
    with mol._mutation_lock:
        uids = tuple(atom._uid for atom in mol._atoms)
        source = _capture_partial_charge_source(mol)
        payload = PartialChargeSet(uids, tuple(values), method, method_version,
                                   parameters, "e", (producer,) * len(uids))
        return _select_partial_charge_result(mol, payload, source, uids)


def _partial_charge_record(mol, rows, *, whole_context):
    """Project historical observations into canonical record row identities."""
    from ._fields import ATOM_SOURCE_CODES

    attachment = mol._partial_charge_result
    if attachment is None:
        return None
    if attachment.source.owner is not mol._owner_token:
        raise RuntimeError("Partial charge record capture requires the current molecular owner")
    payload = attachment.payload
    observed = dict(zip(payload.atom_uids, zip(payload.values, payload.provenance)))
    selected = tuple(observed.get(mol._atoms[row]._uid, (None, ATOM_SOURCE_CODES["default"]))
                     for row in rows)
    marker = attachment.invalidated_by
    if not whole_context and "charge_representation" not in marker:
        marker += ("charge_representation",)
    return (replace(payload, atom_uids=tuple(range(len(rows))),
                    values=tuple(value for value, _ in selected),
                    provenance=tuple(code for _, code in selected)), marker)


def _install_partial_charge_record(mol, result):
    from ._commit import _current_transaction

    transaction = _current_transaction(mol)
    if transaction is not None:
        transaction.touch_results()
    if result is None:
        mol._partial_charge_result = None
        return
    payload, marker = result
    mol._partial_charge_result = _ResultAttachment(
        replace(payload, atom_uids=tuple(atom._uid for atom in mol._atoms)),
        _capture_partial_charge_source(mol), marker,
    )
