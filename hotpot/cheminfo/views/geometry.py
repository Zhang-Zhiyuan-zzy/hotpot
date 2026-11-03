"""Source-bound angle and torsion views with writable constraint projections."""

from ..structure.atom import Atom
from ..structure._commit import _commit_constraints
from .sequence import AtomSeq


class Angle(AtomSeq):
    """
    Represents an angle defined by three atoms in a sequence.

    This class determines the angular relationship between three atoms represented as
    coordinates. It provides functionality to calculate the angle in degrees
    and supports defining constraints for the angular relationship. Extends from
    the AtomSeq class to handle sequences of atoms.

    Attributes:
    constraint: Defines if a constraint is applied to the angle. Default is False.
    """
    def __init__(self, a1: Atom, a2: Atom, a3: Atom):
        super().__init__(a1, a2, a3)

    @property
    def degrees(self) -> float:
        from ..operations.geometry import _angle_degrees
        return _angle_degrees(self)



class Torsion(AtomSeq):
    """
    Represents a torsion or dihedral angle defined by a sequence of four atoms.

    This class models a torsion angle in molecular geometry, which is the angle
    formed by four atoms arranged in a sequence. It inherits from the AtomSeq
    class to represent the ordered sequence of atoms defining the torsion.
    This class also provides properties to compute metrics such as the torsion
    angle in degrees and determine whether the torsion is rotatable.

    Attributes:
        constraint (bool): Indicates if a constraint is applied to the torsion
            angle. Defaults to False.

    Methods:
        degrees (float): Returns the torsion angle in degrees based on the
            coordinates of the atoms in the sequence.

        rotatable (bool): Determines if the torsion angle is rotatable based
            on the properties of the connecting bonds.
    """
    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom):
        super().__init__(a, b, c, d)

    @property
    def degrees(self) -> float:
        from ..operations.geometry import _torsion_degrees
        return _torsion_degrees(self)

    @property
    def rotatable(self) -> bool:
        """
        Returns whether the bond is rotatable.

        The property determines if the associated bond object is rotatable
        based on the characteristics of `bond2`. This is typically used
        to ascertain flexibility or degrees of freedom in molecular
        structures.

        @return: True if the bond is rotatable, False otherwise
        """
        with self.mol._mutation_lock:
            self._check_topology_source()
            return self.bond2.rotatable



def _constraint_property(scope):
    def read(view):
        from ._source import _check_view_source
        with view.mol._mutation_lock:
            _check_view_source(view.mol, view._view_source, view.atoms, view._view_uids)
            return view._view_uids in getattr(view.mol._constraints, scope)

    def write(view, value):
        from ._source import _check_view_source
        with view.mol._mutation_lock:
            _check_view_source(view.mol, view._view_source, view.atoms, view._view_uids)
            entries = ((view.atoms, value),)
            if scope == 'angles':
                _commit_constraints(view.mol, view._view_source[0], angle_entries=entries)
            else:
                _commit_constraints(view.mol, view._view_source[0], torsion_entries=entries)

    return property(read, write)



Angle.constraint = _constraint_property('angles')
Torsion.constraint = _constraint_property('torsions')
