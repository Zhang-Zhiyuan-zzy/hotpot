"""Exact affine and lattice mathematics for the one crystallographic domain."""

from fractions import Fraction
from math import lcm
import ast

__all__ = ()

_IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_ZERO = (Fraction(0),) * 3


def _q(value):
    if isinstance(value, float):
        raise TypeError("CRYSTAL-CONTEXT-INTEGRITY: use exact integer/rational inputs")
    return Fraction(value)


def _vector(values):
    values = tuple(map(_q, values))
    if len(values) != 3:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: translation requires three entries")
    return values


def _matrix(values):
    values = tuple(tuple(map(_q, row)) for row in values)
    if len(values) != 3 or any(len(row) != 3 for row in values):
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: matrix requires shape (3,3)")
    return values


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def _mv(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _mm(first, second):
    columns = tuple(zip(*second))
    return tuple(tuple(_dot(row, column) for column in columns) for row in first)


def _det(matrix):
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _inverse(matrix):
    (a, b, c), (d, e, f), (g, h, i) = matrix
    determinant = _det(matrix)
    if not determinant:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: singular setting matrix")
    return tuple(tuple(Fraction(value, determinant) for value in row) for row in (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    ))


def _integer_rotation(values):
    matrix = _matrix(values)
    if any(value.denominator != 1 for row in matrix for value in row):
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: rotation does not preserve cell Z^3")
    result = tuple(tuple(int(value) for value in row) for row in matrix)
    if abs(_det(result)) != 1:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: rotation is not unimodular")
    _rotation_order(result)
    return result


def _rotation_order(rotation):
    value = _IDENTITY
    for order in range(1, 7):
        value = _mm(value, rotation)
        if value == _IDENTITY:
            if order not in (1, 2, 3, 4, 6):
                break
            return order
    raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: noncrystallographic rotation order")


def _compose(first, second):
    rotation, translation = first
    other_rotation, other_translation = second
    return (_mm(rotation, other_rotation),
            tuple(a + b for a, b in zip(_mv(rotation, other_translation), translation)))


def _invert(operation):
    rotation, translation = operation
    inverse = _inverse(rotation)
    return (tuple(tuple(int(value) for value in row) for row in inverse),
            tuple(-value for value in _mv(inverse, translation)))


def _reduce(operation):
    rotation, translation = operation
    carry = tuple(value.numerator // value.denominator for value in translation)
    return (rotation, tuple(value - image for value, image in zip(translation, carry))), carry


def _canonical(operations):
    values = {_reduce((rotation, translation))[0] for rotation, translation in operations}
    identity = (_IDENTITY, _ZERO)
    return tuple(sorted(values, key=lambda value: (value != identity, value)))


def _validate_group(operations):
    values = frozenset(operations)
    if (_IDENTITY, _ZERO) not in values:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: complete group requires identity")
    for first in operations:
        if _reduce(_invert(first))[0] not in values:
            raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: incomplete group inverse")
        for second in operations:
            if _reduce(_compose(first, second))[0] not in values:
                raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: group is not closed")


def _translation_basis(translations):
    """Positive-diagonal lower column Hermite basis of Z^3 plus translations."""
    denominator = lcm(*(value.denominator for vector in translations for value in vector))
    rows = [[denominator * value for value in row] for row in _IDENTITY]
    rows += [[int(denominator * value) for value in vector] for vector in translations]
    pivot = 0
    for column in range(3):
        while True:
            choices = [index for index in range(pivot, len(rows)) if rows[index][column]]
            smallest = min(choices, key=lambda index: abs(rows[index][column]))
            rows[pivot], rows[smallest] = rows[smallest], rows[pivot]
            for index in range(pivot + 1, len(rows)):
                quotient = rows[index][column] // rows[pivot][column]
                rows[index] = [a - quotient * b for a, b in zip(rows[index], rows[pivot])]
            if not any(row[column] for row in rows[pivot + 1:]):
                break
        if rows[pivot][column] < 0:
            rows[pivot] = [-value for value in rows[pivot]]
        for index in range(pivot):
            quotient = rows[index][column] // rows[pivot][column]
            rows[index] = [a - quotient * b for a, b in zip(rows[index], rows[pivot])]
        pivot += 1
    return tuple(tuple(Fraction(value, denominator) for value in row)
                 for row in zip(*rows[:3]))


def _translation_cosets(generators):
    """Enumerate the finite subgroup of Q^3/Z^3 generated by these vectors."""
    generators = tuple(_reduce((_IDENTITY, vector))[0][1] for vector in generators)
    seen = {_ZERO}
    pending = [_ZERO]
    for current in pending:
        for vector in generators:
            candidate = _reduce((_IDENTITY, tuple(a + b for a, b in zip(current, vector))))[0][1]
            if candidate not in seen:
                seen.add(candidate)
                pending.append(candidate)
    return tuple(sorted(seen))


def _change_frame(operations, basis, origin):
    """Return full new-cell representatives for f_old=basis*f_new+origin."""
    inverse = _inverse(basis)
    translations = tuple(translation for rotation, translation in operations if rotation == _IDENTITY)
    new_lattice = _mm(inverse, _translation_basis(translations))
    cell_in_lattice = _inverse(new_lattice)
    if any(value.denominator != 1 for row in cell_in_lattice for value in row):
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: new cell is not a translation sublattice")
    extra_translations = _translation_cosets(tuple(zip(*inverse)))
    transformed = []
    for rotation, translation in operations:
        new_rotation = _integer_rotation(_mm(_mm(inverse, rotation), basis))
        shift = tuple(a + b - c for a, b, c in zip(_mv(rotation, origin), translation, origin))
        new_translation = _mv(inverse, shift)
        transformed.extend((new_rotation, tuple(a + b for a, b in zip(new_translation, extra)))
                           for extra in extra_translations)
    return _canonical(transformed)


def _parse_expression(expression):
    zero = (Fraction(0),) * 4

    def linear(node):
        if isinstance(node, ast.Name) and node.id in ('x', 'y', 'z'):
            return tuple(Fraction(int(index == ('x', 'y', 'z').index(node.id))) for index in range(4))
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return zero[:3] + (Fraction(node.value),)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return tuple(sign * value for value in linear(node.operand))
        if isinstance(node, ast.BinOp):
            first, second = linear(node.left), linear(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sign = -1 if isinstance(node.op, ast.Sub) else 1
                return tuple(a + sign * b for a, b in zip(first, second))
            if isinstance(node.op, ast.Mult):
                if any(first[:3]) and any(second[:3]):
                    raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: xyz expression is not affine")
                scalar, vector = (first[3], second) if not any(first[:3]) else (second[3], first)
                return tuple(scalar * value for value in vector)
            if isinstance(node.op, ast.Div) and not any(second[:3]) and second[3]:
                return tuple(value / second[3] for value in first)
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: unsupported exact xyz expression")

    return linear(ast.parse(expression.strip(), mode='eval').body)


def _parse_xyz(symbol):
    expressions = str(symbol).split(',')
    if len(expressions) != 3:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: xyz requires three expressions")
    rows = tuple(_parse_expression(expression) for expression in expressions)
    return _integer_rotation(tuple(row[:3] for row in rows)), tuple(row[3] for row in rows)


def _format_xyz(operation):
    expressions = []
    for row, translation in zip(*operation):
        terms = []
        for coefficient, name in zip(row, ('x', 'y', 'z')):
            if coefficient:
                term = name if abs(coefficient) == 1 else f'{abs(coefficient)}*{name}'
                terms.append(('-' if coefficient < 0 else '+') + term)
        if translation:
            terms.append(('-' if translation < 0 else '+') + str(abs(translation)))
        expression = ''.join(terms).lstrip('+')
        expressions.append(expression or '0')
    return ','.join(expressions)


def _kind(operation):
    rotation, translation = operation
    if rotation == _IDENTITY:
        return 'translation' if any(translation) else 'identity'
    if rotation == tuple(tuple(-value for value in row) for row in _IDENTITY):
        return 'inversion'
    order = _rotation_order(rotation)
    power = (_IDENTITY, _ZERO)
    for _ in range(order):
        power = _compose(power, operation)
    if _det(rotation) == 1:
        return 'screw' if any(power[1]) else 'rotation'
    if order == 2:
        return 'glide' if any(power[1]) else 'reflection'
    return 'rotoinversion'
