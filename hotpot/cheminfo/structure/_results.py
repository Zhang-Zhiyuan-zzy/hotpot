"""Finite selected-result ownership and operation-boundary source bindings."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from ._identity import _SourceIdentity, _capture_source, _check_source

__all__ = ()

_T = TypeVar("_T")

_RESULT_ATTRIBUTES = ("_mca_result",)
_REPRESENTATION_FIELDS = frozenset((
    "atomic_number", "formal_charge", "implicit_hydrogens", "is_aromatic",
))
_GEOMETRY_FIELDS = frozenset(("x", "y", "z"))
_MCA_OPERATION = "hotpot.cheminfo.calculator.mca(mol, backend=...)"


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
        domains = (*domains, "mca_representation")
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
