"""Composable search expressions for the public :mod:`hotpot.ds` API.

The classes in this module deliberately depend on the public dataset objects by
protocol rather than by concrete type.  This keeps the search layer usable for
dataset views and third-party adapters without introducing an import cycle.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any


__all__ = [
    "BaseSearcher",
    "DataSearcher",
    "DatasetSearcher",
    "SmartsSearcher",
]


_MISSING = object()
_MULTI_VALUE_TYPES = (list, tuple, set, frozenset)


def _step(value, part):
    """Resolve one path component on mappings, sequences, or objects."""
    if isinstance(value, Mapping):
        return value.get(part, _MISSING)

    if isinstance(value, (list, tuple)) and part.isdigit():
        index = int(part)
        return value[index] if index < len(value) else _MISSING

    try:
        return getattr(value, part)
    except (AttributeError, TypeError):
        return _MISSING


def _resolve_path(value, path):
    """Resolve an exact key or a dotted/``__`` nested field path."""
    exact = _step(value, path)
    if exact is not _MISSING:
        return exact

    parts = path.replace("__", ".").split(".")
    current = value
    for part in parts:
        current = _step(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def _item_info(item):
    if isinstance(item, Mapping):
        return item
    try:
        info = getattr(item, "info")
    except (AttributeError, TypeError):
        return _MISSING
    return info() if callable(info) else info


def _field_value(item, path):
    """Read a public field, preferring the object's JSON-compatible ``info``."""
    info = _item_info(item)
    if info is not _MISSING:
        value = _resolve_path(info, path)
        if value is not _MISSING:
            return value

    value = _resolve_path(item, path)
    if value is not _MISSING:
        return value

    if isinstance(item, str) and path in {
        "name",
        "ds_name",
        "dataset",
        "dataset_name",
    }:
        return item
    return _MISSING


def _values_equal(left, right):
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _matches_value(actual, expected):
    """Apply equality or IN semantics to one resolved field value."""
    if actual is _MISSING:
        return False
    if callable(expected):
        return bool(expected(actual))

    if isinstance(expected, _MULTI_VALUE_TYPES):
        options = tuple(expected)
        if isinstance(actual, _MULTI_VALUE_TYPES):
            return any(
                _values_equal(value, option)
                for value in actual
                for option in options
            )
        return any(_values_equal(actual, option) for option in options)

    if isinstance(actual, _MULTI_VALUE_TYPES):
        return any(_values_equal(value, expected) for value in actual)
    return _values_equal(actual, expected)


class BaseSearcher(ABC):
    """Abstract, composable predicate over either data or dataset objects.

    Subclasses implement :meth:`match` and declare ``scope`` as ``"data"`` or
    ``"dataset"``.  Expressions of the same scope compose with ``&``, ``|``,
    ``^`` and ``~``.  Python does not expose an overloadable unary ``!``
    operator, so :meth:`not_` is its named equivalent.
    """

    scope = None

    @abstractmethod
    def match(self, item):
        """Return whether *item* satisfies this expression."""
        raise NotImplementedError

    def __call__(self, item):
        return self.match(item)

    def search(self, target):
        """Filter a dataset view or an iterable of public objects.

        A data-scoped expression preserves a dataset as a dataset view.  A
        dataset-scoped expression always returns a tuple, including when the
        target is the public datasets catalog.
        """
        if self.scope == "data" and hasattr(target, "_select"):
            filtered = getattr(target, "_filtered", None)
            if callable(filtered):
                return filtered(self.match)
            return target._select(item for item in target if self.match(item))

        if self.scope == "dataset":
            if hasattr(target, "_dataset_objects") and callable(target._dataset_objects):
                candidates = target._dataset_objects()
            elif hasattr(target, "_select"):
                candidates = (target,)
            else:
                candidates = _as_candidates(target)
        else:
            candidates = _as_candidates(target)

        return tuple(item for item in candidates if self.match(item))

    def _combine(self, other, operator):
        if not isinstance(other, BaseSearcher):
            return NotImplemented
        if self.scope != other.scope:
            raise TypeError(
                "Cannot combine data and dataset searchers in one expression"
            )
        return _CompositeSearcher(operator, self, other)

    def __and__(self, other):
        return self._combine(other, "and")

    def __or__(self, other):
        return self._combine(other, "or")

    def __xor__(self, other):
        return self._combine(other, "xor")

    def __invert__(self):
        return _CompositeSearcher("not", self)

    def not_(self):
        """Return the logical negation of this search expression."""
        return ~self

    def __bool__(self):
        raise TypeError(
            "A search expression has no truth value; use .match(item), "
            ".search(target), or the &, |, ^ and ~ operators"
        )


def _as_candidates(target):
    if isinstance(target, (str, bytes, Mapping)):
        return (target,)
    if isinstance(target, Iterable):
        return target
    return (target,)


class _FieldSearcher(BaseSearcher):
    """Shared field-query implementation for public data containers."""

    def __init__(self, **criteria: Any):
        self._criteria = MappingProxyType(dict(criteria))

    @property
    def criteria(self):
        return self._criteria

    def match(self, item):
        return all(
            _matches_value(_field_value(item, path), expected)
            for path, expected in self._criteria.items()
        )

    def __repr__(self):
        arguments = ", ".join(
            "{}={!r}".format(path, expected)
            for path, expected in self._criteria.items()
        )
        return "{}({})".format(type(self).__name__, arguments)


class DataSearcher(_FieldSearcher):
    """Search fields in ``data.info`` using equality and IN semantics."""

    scope = "data"


class DatasetSearcher(_FieldSearcher):
    """Search fields in ``dataset.info`` using equality and IN semantics."""

    scope = "dataset"


class SmartsSearcher(BaseSearcher):
    """Match data records whose molecular structure contains a SMARTS query."""

    scope = "data"

    def __init__(self, smarts: str, *, semantics=None):
        from hotpot.cheminfo.search import Searcher, Substructure

        self.smarts = smarts
        if semantics is None:
            substructure = Substructure.from_smarts(smarts)
        else:
            substructure = Substructure.from_smarts(smarts, semantics=semantics)
        self._searcher = Searcher(substructure)

    def match(self, data):
        try:
            molecule = (
                data
                if hasattr(data, "atom_bond_graph")
                else getattr(data, "molecule")
            )
            if molecule is None:
                return False
            return bool(self._searcher.has_match(molecule))
        except Exception:
            # Missing, unsupported, or unreadable structures are non-matches.
            return False

    def __repr__(self):
        return "SmartsSearcher({!r})".format(self.smarts)


class _CompositeSearcher(BaseSearcher):
    def __init__(self, operator, left, right=None):
        self.operator = operator
        self.left = left
        self.right = right
        self.scope = left.scope

    def match(self, item):
        if self.operator == "not":
            return not self.left.match(item)
        if self.operator == "and":
            return self.left.match(item) and self.right.match(item)
        if self.operator == "or":
            return self.left.match(item) or self.right.match(item)
        return self.left.match(item) != self.right.match(item)

    def __repr__(self):
        if self.operator == "not":
            return "~{!r}".format(self.left)
        symbols = {"and": "&", "or": "|", "xor": "^"}
        return "({!r} {} {!r})".format(
            self.left,
            symbols[self.operator],
            self.right,
        )
