"""Attributed exact reference-setting lookup; no toolkit or mutable group cache."""

import json
from importlib.resources import files

__all__ = ()


def _reference_group(number):
    if not 1 <= number <= 230:
        raise ValueError('CRYSTAL-CONTEXT-INTEGRITY: space-group number must be 1..230')
    data = json.loads(files(__package__).joinpath('data/space_groups.json').read_text())
    return data['groups'][number - 1], data['rotations']
