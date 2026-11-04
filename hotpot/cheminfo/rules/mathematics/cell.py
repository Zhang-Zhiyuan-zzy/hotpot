"""Pure cell mathematics over an oriented Cartesian row basis."""

import numpy as np


__all__ = ()


def _validate_matrix(matrix):
    """Return an independent finite right-handed float64 cell basis."""
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: cell requires a finite (3,3) matrix")
    volume = np.linalg.det(matrix)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: cell requires finite positive signed volume")
    return matrix


def _cell_matrix(parameters):
    """Build canonical rows: a along +x, b in +xy, and positive volume."""
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (6,) or not np.isfinite(parameters).all():
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: cell requires six finite parameters")
    lengths, angles = parameters[:3], parameters[3:]
    if np.any(lengths <= 0) or np.any(angles <= 0) or np.any(angles >= 180):
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: positive lengths and angles between 0 and 180 required")
    angle_sum = angles.sum()
    if angle_sum >= 360 or np.any(2 * angles >= angle_sum):
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: angles do not enclose a positive-volume cell")
    a, b, c = lengths
    alpha, beta, gamma = np.deg2rad(angles)
    ca, cb, cg = np.cos((alpha, beta, gamma))
    sg = np.sin(gamma)
    cy = (ca - cb * cg) / sg
    cz_squared = 1 - cb * cb - cy * cy
    if cz_squared <= 0:
        raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: angles do not enclose a positive-volume cell")
    return _validate_matrix(((a, 0., 0.), (b * cg, b * sg, 0.),
                             (c * cb, c * cy, c * np.sqrt(cz_squared))))


def _matrix_to_params(matrix):
    """Return a,b,c and standard alpha=(b,c), beta=(a,c), gamma=(a,b)."""
    matrix = _validate_matrix(matrix)
    lengths = np.hypot.reduce(matrix, axis=1)
    unit = matrix / lengths[:, None]
    cosines = (np.dot(unit[1], unit[2]), np.dot(unit[0], unit[2]), np.dot(unit[0], unit[1]))
    angles = np.rad2deg(np.arccos(np.clip(cosines, -1., 1.)))
    return tuple(float(value) for value in (*lengths, *angles))


def _check_metric(matrix, rotations):
    """Require every fractional column rotation to preserve the cell metric."""
    matrix = _validate_matrix(matrix)
    metric = matrix @ matrix.T
    for rotation in rotations:
        rotation = np.asarray(rotation, dtype=np.float64)
        if not np.allclose(rotation.T @ metric @ rotation, metric, rtol=1e-10, atol=1e-10):
            raise ValueError("CRYSTAL-CONTEXT-INTEGRITY: rotation is incompatible with the cell metric")


def _replace_parameter(matrix, index, value):
    """Change canonical shape while retaining its rigid Cartesian orientation."""
    matrix = _validate_matrix(matrix)
    parameters = list(_matrix_to_params(matrix))
    orientation = np.linalg.solve(_cell_matrix(parameters), matrix)
    parameters[index] = value
    return _validate_matrix(_cell_matrix(parameters) @ orientation)


def _to_fractional(cartesian, matrix):
    """Convert (...,3) Cartesian rows to independent fractional rows."""
    matrix = _validate_matrix(matrix)
    cartesian = np.asarray(cartesian, dtype=np.float64)
    return np.linalg.solve(matrix.T, cartesian.reshape(-1, 3).T).T.reshape(cartesian.shape)


def _to_cartesian(fractional, matrix):
    """Convert (...,3) fractional rows to independent Cartesian rows."""
    return np.asarray(fractional, dtype=np.float64) @ _validate_matrix(matrix)
