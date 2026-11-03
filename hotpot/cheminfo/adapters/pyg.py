"""Lazy access to the external Torch serialization boundary."""


__all__ = ()


def _torch_save():
    import torch
    return torch.save
