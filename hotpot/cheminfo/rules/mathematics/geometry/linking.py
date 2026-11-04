"""Finite polygonal Gauss linking with exact separation and interval resolution.

Orientation is (p-q).(dp x dq)/(4*pi), as in BGS arXiv:1912.13121v1,
Eq. (1). Segment-pair angles use the unnormalized Eq. (15) formula.
No external implementation, floating-point transcendental or native kernel is used.
"""

from fractions import Fraction
from math import inf, isqrt, nextafter


__all__ = ()
_PRECISION_BITS = 128
_SCALE = 1 << _PRECISION_BITS


def _sub(first, second):
    return tuple(a - b for a, b in zip(first, second))


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def _cross(first, second):
    return (first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0])


def _point_segment_squared(point, first, second):
    edge, offset = _sub(second, first), _sub(point, first)
    parameter = min(Fraction(1), max(Fraction(0), _dot(offset, edge) / _dot(edge, edge)))
    residual = tuple(value - parameter * direction for value, direction in zip(offset, edge))
    return _dot(residual, residual)


def _segment_squared(first, last, other, other_last):
    """Exact constrained quadratic minimum for two nonzero segments."""
    distances = [_point_segment_squared(point, a, b) for point, a, b in (
        (first, other, other_last), (last, other, other_last),
        (other, first, last), (other_last, first, last))]
    u, v, w = _sub(last, first), _sub(other_last, other), _sub(first, other)
    a, b, c, d, e = _dot(u, u), _dot(u, v), _dot(v, v), _dot(u, w), _dot(v, w)
    determinant = a * c - b * b
    if determinant:
        s, t = (b * e - c * d) / determinant, (a * e - b * d) / determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            residual = tuple(value + s * first_direction - t * second_direction
                             for value, first_direction, second_direction in zip(w, u, v))
            distances.append(_dot(residual, residual))
    return min(distances)


def _interval(value):
    numerator = value.numerator * _SCALE
    denominator = value.denominator
    return numerator // denominator, -(-numerator // denominator)


def _add(first, second):
    return first[0] + second[0], first[1] + second[1]


def _negative(value):
    return -value[1], -value[0]


def _difference(first, second):
    return _add(first, _negative(second))


def _multiply(first, second):
    products = tuple(a * b for a in first for b in second)
    return min(products) // _SCALE, -(-max(products) // _SCALE)


def _divide_positive(first, second):
    fractions = tuple((a * _SCALE, b) for a in first for b in second)
    return (min(numerator // denominator for numerator, denominator in fractions),
            max(-(-numerator // denominator) for numerator, denominator in fractions))


def _integer_divide(value, divisor):
    return value[0] // divisor, -(-value[1] // divisor)


def _sqrt(value):
    lower = isqrt(value[0] * _SCALE)
    upper_square = value[1] * _SCALE
    upper = isqrt(upper_square)
    return lower, upper + (upper * upper != upper_square)


def _atan_small(value):
    """Enclose atan(z), 0<=z<=1/2, including the alternating-series tail."""
    argument = _interval(value)
    square = _multiply(argument, argument)
    power, total, index = argument, (0, 0), 0
    while power[1] > 1:
        term = _integer_divide(power, 2 * index + 1)
        total = _add(total, term if index % 2 == 0 else _negative(term))
        power = _multiply(power, square)
        index += 1
    tail = (0, -(-power[1] // (2 * index + 1)))
    return _add(total, tail if index % 2 == 0 else _negative(tail))


def _pi_interval():
    # Machin's identity follows from (5+i)^4 = (239+i)*(2+2i).
    first, second = _atan_small(Fraction(1, 5)), _atan_small(Fraction(1, 239))
    return _difference((16 * first[0], 16 * first[1]), (4 * second[0], 4 * second[1]))


def _atan_positive(value, pi):
    if value > 1:
        return _difference(_integer_divide(pi, 2), _atan_positive(1 / value, pi))
    if value > Fraction(1, 2):
        return _difference(_integer_divide(pi, 4), _atan_small((1 - value) / (1 + value)))
    return _atan_small(value)


def _atan2_positive_y(y, x, pi):
    if x > 0:
        return _atan_positive(y * _SCALE / x, pi)
    if x < 0:
        return _difference(pi, _atan_positive(-y * _SCALE / x, pi))
    return _integer_divide(pi, 2)


def _triangle_angle(first, second, third, pi):
    """Outward enclosure of Eq. (15)'s half-solid-angle atan2."""
    scale = max(abs(value) for vector in (first, second, third) for value in vector)
    a, b, c = tuple(tuple(value / scale for value in vector) for vector in (first, second, third))
    y = _dot(a, _cross(b, c))
    if y == 0:
        # Called only after proving disjoint segments. Their difference convex
        # hull excludes zero; a coplanar triangle therefore has zero solid angle.
        return 0, 0
    lengths = tuple(_sqrt(_interval(_dot(vector, vector))) for vector in (a, b, c))
    x = _multiply(_multiply(lengths[0], lengths[1]), lengths[2])
    for dot, length in ((_dot(b, c), lengths[0]), (_dot(c, a), lengths[1]), (_dot(a, b), lengths[2])):
        x = _add(x, _multiply(_interval(dot), length))
    lower = _atan2_positive_y(abs(y), x[1], pi)[0]
    upper = _atan2_positive_y(abs(y), x[0], pi)[1]
    return (lower, upper) if y > 0 else (-upper, -lower)


def _pair_angle(first, last, other, other_last, pi):
    alpha, beta = _sub(other, first), _sub(other, last)
    gamma, omega = _sub(other_last, last), _sub(other_last, first)
    if _dot(alpha, _cross(beta, gamma)) == 0:
        return 0, 0
    return _difference(_triangle_angle(alpha, beta, gamma, pi),
                       _triangle_angle(alpha, omega, gamma, pi))


def _simple_curve_issue(points, squared_tolerance, curve):
    count = len(points)
    if count < 3:
        return (("reason", "fewer_than_three_vertices"), ("curve_index", curve))
    for edge in range(count):
        direction = _sub(points[(edge + 1) % count], points[edge])
        if _dot(direction, direction) <= squared_tolerance:
            return (("reason", "edge_length_within_tolerance"), ("curve_index", curve), ("edge_index", edge))
    for vertex in range(count):
        incoming = _sub(points[vertex], points[vertex - 1])
        outgoing = _sub(points[(vertex + 1) % count], points[vertex])
        if _cross(incoming, outgoing) == (0, 0, 0) and _dot(incoming, outgoing) < 0:
            return (("reason", "adjacent_edges_overlap"), ("curve_index", curve), ("vertex_index", vertex))
    for first in range(count):
        for second in range(first + 1, count):
            if second - first in (1, count - 1):
                continue
            if _segment_squared(points[first], points[(first + 1) % count],
                                points[second], points[(second + 1) % count]) <= squared_tolerance:
                return (("reason", "nonadjacent_edges_within_tolerance"),
                        ("curve_index", curve), ("edge_indices", (first, second)))
    return ()


def _polygon_linking(first, second, *, tolerance):
    """Return a resolved integer or None and immutable numerical measurements.

    Input binary64 coordinates are treated as exact. Separation/simple-curve
    predicates use exact rational squared distances; each edge must exceed the
    caller's Angstrom tolerance, as must nonadjacent self and intercurve distances.
    Collinear forward subdivision is allowed; adjacent retracing is not.
    All angle arithmetic uses outward 128-bit dyadic intervals, including square
    roots, arctangent series tails and pi. A value is returned only if the final
    Gauss interval contains exactly one integer. Zero is not an unlink certificate.
    """
    curves = tuple(tuple(tuple(Fraction(float(value)) for value in point) for point in points)
                   for points in (first, second))
    squared_tolerance = Fraction(float(tolerance)) ** 2
    for curve, points in enumerate(curves):
        issue = _simple_curve_issue(points, squared_tolerance, curve)
        if issue:
            return None, issue
    p, q = curves
    for first_edge in range(len(p)):
        for second_edge in range(len(q)):
            if _segment_squared(p[first_edge], p[(first_edge + 1) % len(p)],
                                q[second_edge], q[(second_edge + 1) % len(q)]) <= squared_tolerance:
                return None, (("reason", "intercurve_edges_within_tolerance"),
                              ("edge_indices", (first_edge, second_edge)))
    pi, total = _pi_interval(), (0, 0)
    for first_edge in range(len(p)):
        for second_edge in range(len(q)):
            total = _add(total, _pair_angle(p[first_edge], p[(first_edge + 1) % len(p)],
                                            q[second_edge], q[(second_edge + 1) % len(q)], pi))
    gauss = _divide_positive(total, (2 * pi[0], 2 * pi[1]))
    lower = nextafter(gauss[0] / _SCALE, -inf)
    upper = nextafter(gauss[1] / _SCALE, inf)
    estimate = (gauss[0] + gauss[1]) / (2 * _SCALE)
    error = nextafter(max(estimate - lower, upper - estimate), inf)
    measurements = (("gauss_lower", lower), ("gauss_upper", upper),
                    ("gauss_estimate", estimate), ("gauss_error_bound", error))
    first_integer, last_integer = -(-gauss[0] // _SCALE), gauss[1] // _SCALE
    if first_integer == last_integer:
        return first_integer, measurements
    return None, (("reason", "numerical_interval_does_not_resolve_one_integer"), *measurements)
