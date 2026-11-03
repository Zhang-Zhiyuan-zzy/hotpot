"""Existing declaration objects for exact eager carrier annotations and defaults.

These fixed accessors perform no calculation and import no concrete carriers.
Scientific carrier calls enter their named operations separately.
"""

__all__ = ()


def _geometry_types():
    from ..rules.mathematics import geometry
    return geometry


def _geometry_reports():
    from ..adapters.structure import geometry
    return geometry


def _graph_defaults():
    from ..rules.mathematics import graph
    return graph


def _forcefield_contracts():
    from . import forcefields
    return forcefields
