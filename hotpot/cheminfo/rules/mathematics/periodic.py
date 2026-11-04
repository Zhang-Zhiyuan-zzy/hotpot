"""Pure half-open projection, image transport and periodic graph arithmetic."""

from collections import deque
from fractions import Fraction
from itertools import product
from math import floor, isqrt, prod

import numpy as np


__all__ = ()


def _wrap_fractional(coordinates):
    """Wrap prevalidated (N,3) rows; shifts are immutable Python-int triples.

    Snap only within an absolute 1e-12 of an integer before floor. The returned
    float64 array is independent, and wrapped + shifts reconstructs the snapped
    input. No coordinates are identified or deduplicated.
    """
    wrapped = np.array(coordinates, dtype=np.float64, copy=True)
    nearest = np.rint(wrapped)
    snap = np.abs(wrapped - nearest) <= 1e-12
    wrapped[snap] = nearest[snap]
    shifts = tuple(tuple(floor(value) for value in row) for row in wrapped)
    wrapped -= np.asarray(shifts, dtype=np.float64).reshape(wrapped.shape)
    return wrapped, shifts


def _image_after_wrap(rotation, image, first_shift, second_shift):
    """Transport one known directed image by R*s + w_second - w_first."""
    return tuple(
        sum(int(rotation[row][column]) * int(image[column]) for column in range(3))
        + int(second_shift[row]) - int(first_shift[row])
        for row in range(3)
    )


def _finite_component_lifts(atom_count, endpoints, images):
    """Return stable topology components and exact finite-lift diagnostics.

    Rows and diagnostic edge indices follow input order. Lift triples align with
    rows, with the first row at zero. Unknown edges or a nonzero cycle yield no
    finite lift. Inconsistent edges disagree with deterministic breadth-first
    known-edge spanning-forest potentials; unknown edges never set a potential.
    """
    endpoints = tuple(endpoints)
    images = tuple(None if image is None else tuple(map(int, image)) for image in images)
    adjacency = [[] for _ in range(atom_count)]
    for (first, second), image in zip(endpoints, images):
        adjacency[first].append((second, image))
        adjacency[second].append((first, None if image is None else tuple(-value for value in image)))

    component_ids = [None] * atom_count
    component_count = 0
    for root in range(atom_count):
        if component_ids[root] is not None:
            continue
        component_ids[root] = component_count
        pending = deque((root,))
        while pending:
            current = pending.popleft()
            for neighbor, _ in adjacency[current]:
                if component_ids[neighbor] is None:
                    component_ids[neighbor] = component_count
                    pending.append(neighbor)
        component_count += 1

    potentials = [None] * atom_count
    for root in range(atom_count):
        if potentials[root] is not None:
            continue
        potentials[root] = (0, 0, 0)
        pending = deque((root,))
        while pending:
            current = pending.popleft()
            for neighbor, image in adjacency[current]:
                if image is not None and potentials[neighbor] is None:
                    potentials[neighbor] = tuple(a + b for a, b in zip(potentials[current], image))
                    pending.append(neighbor)

    rows = [[] for _ in range(component_count)]
    unknown = [[] for _ in range(component_count)]
    inconsistent = [[] for _ in range(component_count)]
    for row, component in enumerate(component_ids):
        rows[component].append(row)
    for edge, ((first, second), image) in enumerate(zip(endpoints, images)):
        component = component_ids[first]
        if image is None:
            unknown[component].append(edge)
        elif potentials[second] != tuple(a + b for a, b in zip(potentials[first], image)):
            inconsistent[component].append(edge)
    return tuple(
        (tuple(members), None if missing or conflicts else tuple(potentials[row] for row in members),
         tuple(missing), tuple(conflicts))
        for members, missing, conflicts in zip(rows, unknown, inconsistent)
    )


def _supercell_edges(atom_count, endpoints, images, repeats):
    """Replicate known directed edges in source-edge then lexicographic k order.

    Atom rows use source_atom * prod(repeats) + lex_index(k). Image arithmetic
    uses Python integers; callers reject any expansion requiring unknown images.
    """
    repeats = tuple(map(int, repeats))
    copies = tuple(product(*(range(repeat) for repeat in repeats)))
    copy_count = prod(repeats)
    edges = []
    for edge, ((first, second), image) in enumerate(zip(endpoints, images)):
        image = tuple(map(int, image))
        for copy_index, copy_image in enumerate(copies):
            target = tuple((value + shift) % repeat for value, shift, repeat in zip(copy_image, image, repeats))
            carry = tuple((value + shift) // repeat for value, shift, repeat in zip(copy_image, image, repeats))
            target_index = (target[0] * repeats[1] + target[1]) * repeats[2] + target[2]
            edges.append((int(first) * copy_count + copy_index, int(second) * copy_count + target_index,
                          carry, edge, copy_image))
    return tuple(edges)


def _integer_image_box(delta, inverse_column_norms, radius):
    """Finite metric enclosure, in row-basis fractional coordinates.

    Outward nextafter only retains floating-boundary candidates; callers still
    compare their actual Cartesian distances against the requested radius.
    """
    widths = float(radius) * np.asarray(inverse_column_norms)
    lower = np.ceil(np.nextafter(-np.asarray(delta) - widths, -np.inf))
    upper = np.floor(np.nextafter(-np.asarray(delta) + widths, np.inf))
    return product(*(range(int(first), int(last) + 1) for first, last in zip(lower, upper)))


def _lattice_images(delta, matrix, radius):
    """All integer images within radius, with no fixed-stencil assumption."""
    basis = np.asarray(matrix, dtype=np.float64)
    inverse_norms = np.linalg.norm(np.linalg.inv(basis), axis=0)
    return tuple(image for image in _integer_image_box(delta, inverse_norms, radius)
                 if np.linalg.norm((np.asarray(delta) + image) @ basis) <= radius)


_CARTESIAN_NEIGHBORS = tuple(product((-1, 0, 1), repeat=3))


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second))


def _cross(first, second):
    return (first[1]*second[2] - first[2]*second[1],
            first[2]*second[0] - first[0]*second[2],
            first[0]*second[1] - first[1]*second[0])


def _metric_width_squared(column, squared_cutoff):
    return squared_cutoff * _dot(column, column)


def _sqrt_upper(value):
    """Outward rational root enclosure; absolute error is at most 2**-128."""
    scaled = value.numerator << 256
    root = isqrt(scaled // value.denominator)
    return Fraction(root + (root*root*value.denominator != scaled), 1 << 128)


def _distance(squared):
    """Correctly round an exact squared distance to binary64."""
    if squared == 0:
        return 0.0
    shift = 128 - (squared.numerator.bit_length() - squared.denominator.bit_length()) // 2
    if shift >= 0:
        root = isqrt((squared.numerator << (2*shift)) // squared.denominator)
        scale = Fraction(1, 1 << shift)
    else:
        root = isqrt(squared.numerator // (squared.denominator << (-2*shift)))
        scale = Fraction(1 << -shift)
    lower, upper = float(root * scale), float((root + 1) * scale)
    if lower == upper:
        return lower
    midpoint = (Fraction(lower) + Fraction(upper)) / 2
    if squared < midpoint*midpoint:
        return lower
    if squared > midpoint*midpoint:
        return upper
    return float(midpoint)


def _periodic_pairs(coordinates, matrix, cutoff, *, include_self_images=False):
    """Return stable (i,j,image,distance) rows, i<j plus optional self images.

    Binary64 input values are treated as exact. A returned image translates j
    in the supplied, unwrapped frame. For self images the first nonzero image
    entry is positive; zero self images are excluded. Rows are sorted by
    (i,j,image), and the cutoff comparison is exact with no widening.

    One rational inverse, one fractional wrap, and one Cartesian bin index are
    prepared per call. Inverse-column Euclidean widths bound images per query
    row, never per row pair. The 27 neighboring *Cartesian bins* have side equal
    to cutoff; they are not a stencil of fractional lattice images. A zero
    cutoff uses unit bins while still comparing exact zero distance.
    """
    points = tuple(tuple(Fraction(float(value)) for value in point) for point in coordinates)
    if not points or (len(points) == 1 and not include_self_images):
        return ()
    basis = tuple(tuple(Fraction(float(value)) for value in row) for row in matrix)
    cofactor_columns = (_cross(basis[1], basis[2]), _cross(basis[2], basis[0]),
                        _cross(basis[0], basis[1]))
    determinant = _dot(basis[0], cofactor_columns[0])
    inverse_columns = tuple(tuple(value / determinant for value in column)
                            for column in cofactor_columns)
    fractional = tuple(tuple(_dot(point, column) for column in inverse_columns) for point in points)
    shifts = tuple(tuple(floor(value) for value in point) for point in fractional)
    wrapped_fractional = tuple(tuple(value - shift for value, shift in zip(point, row_shift))
                               for point, row_shift in zip(fractional, shifts))
    wrapped = tuple(tuple(point[axis] - sum(shift[k]*basis[k][axis] for k in range(3))
                          for axis in range(3)) for point, shift in zip(points, shifts))
    minimum = tuple(min(point[axis] for point in wrapped_fractional) for axis in range(3))
    maximum = tuple(max(point[axis] for point in wrapped_fractional) for axis in range(3))
    radius = Fraction(float(cutoff))
    squared_cutoff = radius*radius
    widths = tuple(_sqrt_upper(_metric_width_squared(column, squared_cutoff))
                   for column in inverse_columns)
    bin_width = radius if radius else Fraction(1)
    bins = {}
    for first, point in enumerate(wrapped):
        key = tuple(floor(value / bin_width) for value in point)
        bins.setdefault(key, []).append(first)
    translations = {}
    rows = []
    for second, (point, fraction) in enumerate(zip(wrapped, wrapped_fractional)):
        bounds = tuple(range(-floor(-(low - value - width)), floor(high - value + width) + 1)
                       for low, high, value, width in zip(minimum, maximum, fraction, widths))
        for image in product(*bounds):
            if image not in translations:
                translations[image] = tuple(sum(image[k]*basis[k][axis] for k in range(3))
                                            for axis in range(3))
            placed = tuple(value + step for value, step in zip(point, translations[image]))
            key = tuple(floor(value / bin_width) for value in placed)
            for offset in _CARTESIAN_NEIGHBORS:
                nearby = tuple(value + step for value, step in zip(key, offset))
                for first in bins.get(nearby, ()):
                    if first > second or (first == second and
                            (not include_self_images or image <= (0, 0, 0))):
                        continue
                    delta = tuple(value - origin for value, origin in zip(placed, wrapped[first]))
                    squared = _dot(delta, delta)
                    if squared <= squared_cutoff:
                        actual_image = tuple(value + left - right for value, left, right
                                             in zip(image, shifts[first], shifts[second]))
                        rows.append((first, second, actual_image, _distance(squared)))
    return tuple(sorted(rows, key=lambda row: row[:3]))
