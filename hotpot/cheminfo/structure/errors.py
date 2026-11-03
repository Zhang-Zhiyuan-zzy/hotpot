"""Molecule ownership and membership errors."""

class NotInSameMolecule(Exception):
    """ The exception is toggled when two objects are expected in a same Molecule, but not """
    def __init__(self, obj1, obj2):
        self.obj1 = obj1
        self.obj2 = obj2
        self.message = f"The {type(obj1).__name__} and {type(obj2).__name__} must belong to the same Molecule"
        super().__init__(self.message)


class ObjNotInMolecule(Exception):
    """ The exception is toggled when the given object not in a specific Molecule. """
    def __init__(self, obj, mol):
        self.obj = obj
        self.mol = mol
        self.message = f"The {type(obj).__name__} is not a current member of the Molecule"
        super().__init__(self.message)
