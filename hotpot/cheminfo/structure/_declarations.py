"""Finite completion of the existing quoted carrier annotation namespaces."""

__all__ = ()


def _complete_type_names():
    # These are concrete leaf modules, already completed by structure.__init__.
    from . import molecule, atom, bond
    from ..views import geometry, pairs, sequence, topology

    molecule.Atom = atom.Atom
    molecule.Bond = bond.Bond
    molecule.Angle = geometry.Angle
    molecule.Torsion = geometry.Torsion
    molecule.Ring = topology.Ring
    molecule.JointRing = topology.JointRing
    molecule.AtomPairs = pairs.AtomPairs
    atom.Bond = bond.Bond
    sequence.Bond = bond.Bond
