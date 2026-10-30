"""Lock-protected access to the transitional installed stereo payload."""

__all__ = ()


def _get_stereo_payload(mol):
    """Return the exact immutable object currently installed on this owner."""

    with mol._mutation_lock:
        return mol._stereo_payload
