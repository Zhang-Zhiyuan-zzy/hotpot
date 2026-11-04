"""Immutable space-group context and complete exact affine operations."""

from abc import ABC, abstractmethod
from fractions import Fraction
import operator

from ..rules.mathematics import symmetry as _math
from ..reference.space_groups import _reference_group

__all__ = ('SpaceGroup', 'SymmUnit')


def _reference(number):
    entry, rotations = _reference_group(number)
    operations = tuple((tuple(tuple(rotations[index][offset:offset + 3]) for offset in (0, 3, 6)),
                        tuple(Fraction(value, 24) for value in translation))
                       for index, translation, _symbol in entry['operations'])
    metadata = tuple(entry[name] for name in ('hermann_mauguin', 'hall', 'crystal_system', 'point_group'))
    return operations, metadata


def _operation(value):
    if isinstance(value, SymmUnit):
        rotation, translation = value.R, value.t
    elif isinstance(value, str):
        return _math._parse_xyz(value)
    else:
        rotation, translation = value
    return _math._integer_rotation(rotation), _math._vector(translation)


def _new_group(operations, number, setting, metadata):
    group = object.__new__(SpaceGroup)
    for name, value in (('_operations', operations), ('_number', number),
                        ('_setting', setting), ('_metadata', metadata)):
        object.__setattr__(group, name, value)
    return group


class SpaceGroup:
    """One exact immutable ordinary 3D space group in a declared cell setting."""

    __slots__ = ('_operations', '_number', '_setting', '_metadata')

    def __init__(self, number: int):
        if hasattr(self, '_operations'):
            raise AttributeError('SpaceGroup is immutable')
        number = operator.index(number)
        operations, metadata = _reference(number)
        # The shipped complete table is qualified at release. Anonymous/external
        # tables are checked at their construction boundary in from_operations.
        for name, value in (('_operations', operations), ('_number', number),
                            ('_setting', (number, _math._IDENTITY, _math._ZERO)),
                            ('_metadata', metadata)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError('SpaceGroup is immutable')

    def __delattr__(self, name):
        raise AttributeError('SpaceGroup is immutable')

    @classmethod
    def from_operations(cls, operators, *, reference_number=None, to_reference=None):
        operations = _math._canonical(tuple(map(_operation, operators)))
        _math._validate_group(operations)
        if (reference_number is None) != (to_reference is None):
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: reference number and setting map are paired')
        setting = None
        metadata = (None,) * 4
        if reference_number is not None:
            reference_number = operator.index(reference_number)
            basis, origin = to_reference
            basis, origin = _math._matrix(basis), _math._vector(origin)
            inverse = _math._inverse(basis)
            projected = _math._change_frame(operations, inverse,
                                            tuple(-value for value in _math._mv(inverse, origin)))
            reference, reference_metadata = _reference(reference_number)
            if projected != _math._canonical(reference):
                raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: setting map does not match reference group')
            setting = reference_number, basis, origin
            metadata = (reference_metadata if basis == _math._IDENTITY and origin == _math._ZERO
                        else (None, None, *reference_metadata[2:]))
        return _new_group(operations, reference_number, setting, metadata)

    @property
    def number(self):
        return self._number

    @property
    def hermann_mauguin(self):
        return self._metadata[0]

    @property
    def hall(self):
        return self._metadata[1]

    @property
    def setting(self):
        return self._setting

    @property
    def crystal_system(self):
        return self._metadata[2]

    @property
    def point_group(self):
        return self._metadata[3]

    @property
    def symm_units(self):
        return tuple(_AffineSymmUnit(rotation, translation, self)
                     for rotation, translation in self._operations)

    @property
    def centering_translations(self):
        return tuple(translation for rotation, translation in self._operations
                     if rotation == _math._IDENTITY)

    @property
    def translation_basis(self):
        return _math._translation_basis(self.centering_translations)

    @property
    def point_rotations(self):
        return tuple(dict.fromkeys(rotation for rotation, _ in self._operations))

    @property
    def cell_order(self):
        return len(self._operations)

    @property
    def point_group_order(self):
        return len(self.point_rotations)

    def identity(self):
        return _AffineSymmUnit(_math._IDENTITY, _math._ZERO, self)

    def lift(self, index: int, image=(0, 0, 0)):
        rotation, translation = self._operations[operator.index(index)]
        image = tuple(map(operator.index, image))
        if len(image) != 3:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: image requires three integers')
        return _AffineSymmUnit(rotation, tuple(value + shift for value, shift in zip(translation, image)), self)

    def contains(self, operation):
        return (isinstance(operation, SymmUnit) and operation.context == self
                and _math._reduce((operation.R, operation.t))[0] in self._operations)

    def __contains__(self, operation):
        return self.contains(operation)

    def change_setting(self, P, p):
        basis, origin = _math._matrix(P), _math._vector(p)
        operations = _math._change_frame(self._operations, basis, origin)
        setting = None
        metadata = (None, None, *self._metadata[2:])
        if self.setting is not None:
            number, previous_basis, previous_origin = self.setting
            setting = (number, _math._mm(previous_basis, basis),
                       tuple(a + b for a, b in zip(_math._mv(previous_basis, origin), previous_origin)))
            if setting[1:] == (_math._IDENTITY, _math._ZERO):
                _, metadata = _reference(number)
        return _new_group(operations, self.number, setting, metadata)

    def __eq__(self, other):
        if not isinstance(other, SpaceGroup):
            return NotImplemented
        return (self._number, self._setting, self._operations) == (other._number, other._setting, other._operations)

    def __hash__(self):
        return hash((self._number, self._setting, self._operations))

    def __repr__(self):
        return f'SpaceGroup(number={self.number!r}, cell_order={self.cell_order})'

    def __reduce__(self):
        return _new_group, (self._operations, self._number, self._setting, self._metadata)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class SymmUnit(ABC):
    """A complete exact element {R|t} of one immutable SpaceGroup context."""

    __slots__ = ()

    @property
    @abstractmethod
    def R(self):
        """Exact integer matrix in the context's fractional convention."""

    @property
    @abstractmethod
    def t(self):
        """Full exact translation, including the integer lattice carry."""

    @property
    @abstractmethod
    def context(self):
        """The sole immutable group/setting owner."""

    @classmethod
    def from_affine(cls, R, t, *, context):
        rotation, translation = _math._integer_rotation(R), _math._vector(t)
        value = _AffineSymmUnit(rotation, translation, context)
        if not context.contains(value):
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: affine operation is outside the supplied group')
        return value

    @classmethod
    def from_xyz(cls, symbol: str, *, context):
        return cls.from_affine(*_math._parse_xyz(symbol), context=context)

    @classmethod
    def identity(cls, *, context):
        return context.identity()

    @property
    def xyz(self):
        return _math._format_xyz((self.R, self.t))

    @property
    def kind(self):
        return _math._kind((self.R, self.t))

    def operate(self, fractional):
        import numpy as np
        values = np.asarray(fractional, dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: fractional coordinates require (...,3)')
        return values @ np.asarray(self.R, dtype=np.float64).T + np.asarray(self.t, dtype=np.float64)

    def compose(self, other):
        if not isinstance(other, SymmUnit) or self.context != other.context:
            raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: composition requires equal group settings')
        return _AffineSymmUnit(*_math._compose((self.R, self.t), (other.R, other.t)), self.context)

    def __matmul__(self, other):
        return self.compose(other)

    def inverse(self):
        return _AffineSymmUnit(*_math._invert((self.R, self.t)), self.context)

    def __invert__(self):
        return self.inverse()

    def power(self, n: int):
        n = operator.index(n)
        value = self if n >= 0 else self.inverse()
        n = abs(n)
        result = self.context.identity()
        while n:
            if n & 1:
                result = result @ value
            value = value @ value
            n >>= 1
        return result

    def __pow__(self, n):
        return self.power(n)

    def reduce_to_cell(self):
        (rotation, translation), carry = _math._reduce((self.R, self.t))
        return _AffineSymmUnit(rotation, translation, self.context), carry

    def __eq__(self, other):
        if not isinstance(other, SymmUnit):
            return NotImplemented
        return (self.R, self.t, self.context) == (other.R, other.t, other.context)

    def __hash__(self):
        return hash((self.R, self.t, self.context))

    def __repr__(self):
        return f'SymmUnit({self.xyz!r}, context={self.context!r})'

    def __reduce__(self):
        return _AffineSymmUnit, (self.R, self.t, self.context)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _AffineSymmUnit(SymmUnit):
    __slots__ = ('_R', '_t', '_context')

    def __init__(self, R, t, context):
        if hasattr(self, '_R'):
            raise AttributeError('SymmUnit is immutable')
        object.__setattr__(self, '_R', R)
        object.__setattr__(self, '_t', t)
        object.__setattr__(self, '_context', context)

    def __setattr__(self, name, value):
        raise AttributeError('SymmUnit is immutable')

    def __delattr__(self, name):
        raise AttributeError('SymmUnit is immutable')

    @property
    def R(self):
        return self._R

    @property
    def t(self):
        return self._t

    @property
    def context(self):
        return self._context
