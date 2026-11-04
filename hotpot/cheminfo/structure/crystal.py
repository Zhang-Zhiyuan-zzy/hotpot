"""One oriented cell, one group association and its contained Molecule facts."""

from ._crystal import _read_crystal_record, _public_relations


class Crystal:
    def __init__(self, a, b, c, alpha, beta, gamma, *, molecule=None,
                 space_group=None, site_relations=None, bond_images=None):
        from ..operations.crystal import _initialize_crystal, _cell_matrix
        _initialize_crystal(self, _cell_matrix((a, b, c, alpha, beta, gamma)),
                            molecule, space_group, site_relations, bond_images)

    @staticmethod
    def _matrix_to_params(matrix):
        from ..rules.mathematics.cell import _matrix_to_params
        return _matrix_to_params(matrix)

    @classmethod
    def from_matrix(cls, matrix, *, molecule=None, space_group=None,
                    site_relations=None, bond_images=None):
        from ..operations.crystal import _initialize_crystal
        value = object.__new__(cls)
        _initialize_crystal(value, matrix, molecule, space_group, site_relations, bond_images)
        return value

    @classmethod
    def from_cif(cls, source, *, block=None):
        from ..operations.crystal_io import _from_cif
        return _from_cif(cls, source, block)

    def to_cif(self):
        from ..operations.crystal_io import _to_cif
        return _to_cif(self)

    @property
    def molecule(self):
        return self._molecule

    @property
    def matrix(self):
        import numpy as np
        return np.array(_read_crystal_record(self)[1], dtype=np.float64)

    @property
    def volume(self):
        from ..operations.crystal import _volume
        return _volume(self.matrix)

    @property
    def space_group(self):
        return _read_crystal_record(self)[2]

    @property
    def symm_units(self):
        group = self.space_group
        return None if group is None else group.symm_units

    @property
    def site_relations(self):
        return _public_relations(self)

    @property
    def bond_images(self):
        return _read_crystal_record(self)[4]

    @property
    def representation(self):
        return _read_crystal_record(self)[5]

    @property
    def image_map(self):
        return _read_crystal_record(self)[7]

    def set_cell(self, matrix, *, preserve='cartesian'):
        from ..operations.crystal import _set_cell
        _set_cell(self, matrix, preserve)

    def set_space_group(self, space_group, *, site_relations=None):
        from ..operations.crystal import _set_space_group
        _set_space_group(self, space_group, site_relations)

    def set_site_relations(self, relations):
        from ..operations.crystal import _set_site_relations
        _set_site_relations(self, relations)

    def to_fractional(self, cartesian):
        from ..operations.crystal import _to_fractional
        return _to_fractional(cartesian, self.matrix)

    def to_cartesian(self, fractional):
        from ..operations.crystal import _to_cartesian
        return _to_cartesian(fractional, self.matrix)

    def operate(self, operation):
        from ..operations.crystal import _operate_crystal
        return _operate_crystal(self, operation)

    def unit_cell(self):
        from ..operations.crystal_generation import _unit_cell_crystal
        return _unit_cell_crystal(self)

    def unit_molecules(self):
        from ..operations.crystal_generation import _finite_molecules
        return _finite_molecules(self)

    def expand(self, repeats):
        from ..operations.crystal_generation import _expand_crystal
        return _expand_crystal(self, repeats)

    def same_molecule_candidates(self, *, tolerance):
        from ..operations.crystal_matching import _same_molecule_candidates
        return _same_molecule_candidates(self, tolerance)

    def molecular_symmetry(self, molecule=None, *, tolerance):
        from ..operations.crystal_matching import _molecular_symmetry
        return _molecular_symmetry(self, molecule, tolerance)

    def linking_number(self, first_cycle, second_cycle, *, tolerance):
        from ..operations.crystal_geometry import _linking_number
        return _linking_number(self, first_cycle, second_cycle, tolerance)

    def geometry_findings(self, *, coincidence_tolerance, close_contact_cutoff,
                          ring_scope='full_graph', max_ring_size=None):
        from ..operations.crystal_geometry import _geometry_findings
        return _geometry_findings(self, coincidence_tolerance, close_contact_cutoff,
                                  ring_scope, max_ring_size)

    def copy(self):
        from ..operations.crystal import _copy_crystal
        return _copy_crystal(self)

    def __reduce_ex__(self, protocol):
        from ._crystal import _restore_active_crystal, _restore_standalone_crystal
        extra = {key: value for key, value in self.__dict__.items() if key not in ('_state', '_molecule')}
        if self._molecule is None:
            return _restore_standalone_crystal, (type(self), _read_crystal_record(self)), extra
        return _restore_active_crystal, (self._molecule,), extra


def _parameter(index):
    def read(crystal):
        from ..operations.crystal import _matrix_to_params
        return _matrix_to_params(_read_crystal_record(crystal)[1])[index]

    def write(crystal, value):
        from ..operations.crystal import _set_parameter
        _set_parameter(crystal, index, value)

    return property(read, write)


for _index, _name in enumerate(('a', 'b', 'c', 'alpha', 'beta', 'gamma')):
    setattr(Crystal, _name, _parameter(_index))
del _index, _name
