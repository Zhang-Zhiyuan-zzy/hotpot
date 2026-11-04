"""Pure half-open projection, image transport and periodic graph arithmetic."""

from collections import deque
from itertools import product
from math import floor, prod

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
