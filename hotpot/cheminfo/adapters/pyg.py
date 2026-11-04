"""Lazy Torch/PyG boundary for molecular transport and graph serialization."""

__all__ = ()


def _torch_save():
    import torch
    return torch.save


def _encode_molecule(record, details, available):
    from ._pyg_molecule import encode
    return encode(record, details, available)


def _decode_molecule(data, field_map):
    from ._pyg_molecule import decode
    return decode(data, field_map)
