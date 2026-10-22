#include <hotpot/native/molecule_buffer_view.hpp>

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace native = hotpot::native;


template <typename Callback>
void assert_invalid(Callback callback) {
    bool raised = false;
    try {
        callback();
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    assert(raised);
}


std::uint64_t fields(std::initializer_list<native::MoleculeField> values) {
    std::uint64_t mask = 0;
    for (const auto value : values) {
        mask |= native::molecule_field_mask(value);
    }
    return mask;
}


void test_array_view() {
    static_assert(std::is_trivially_copyable_v<native::ArrayView<int>>);
    static_assert(
        static_cast<std::uint32_t>(native::StereoAnchorKind::ATOM) == 1
    );
    static_assert(
        static_cast<std::uint32_t>(native::StereoAnchorKind::BOND) == 2
    );
    static_assert(
        static_cast<std::uint32_t>(native::StereoAnchorKind::AXIS) == 3
    );
    const std::array<int, 3> values = {{2, 4, 8}};
    const native::ArrayView<int> view(values);
    assert(view.data() == values.data());
    assert(view.size() == 3);
    assert(view[2] == 8);
}


void test_field_presence_and_generic_validation() {
    native::OwnedMoleculeData molecule;
    molecule.atom_count = 2;
    molecule.bond_count = 1;
    molecule.fields = fields({
        native::MoleculeField::ATOMIC_NUMBERS,
        native::MoleculeField::FORMAL_CHARGES,
        native::MoleculeField::BOND_INDICES,
        native::MoleculeField::BOND_KINDS,
    });
    molecule.atomic_numbers = {6, 7};
    molecule.formal_charges = {0, 0};
    molecule.bond_indices = {{{0, 1}}};
    molecule.bond_kinds = {native::BondKind::DATIVE};

    molecule.validate();
    const auto view = molecule.view();
    assert(view.has(native::MoleculeField::ATOMIC_NUMBERS));
    assert(!view.has(native::MoleculeField::COORDINATES));
    assert(view.coordinates.empty());
    assert(view.bond_kinds[0] == native::BondKind::DATIVE);

    molecule.partial_charges = {0.0, 0.0};
    assert_invalid([&]() { molecule.validate(); });
    molecule.partial_charges.clear();

    molecule.fields |= native::molecule_field_mask(
        native::MoleculeField::COORDINATES
    );
    assert_invalid([&]() { molecule.validate(); });
}


void test_empty_present_field_differs_from_absence() {
    native::OwnedMoleculeData molecule;
    molecule.fields = fields({native::MoleculeField::ATOMIC_NUMBERS});
    molecule.validate();
    assert(molecule.view().has(native::MoleculeField::ATOMIC_NUMBERS));

    molecule.fields = 0;
    molecule.validate();
    assert(!molecule.view().has(native::MoleculeField::ATOMIC_NUMBERS));
}


void test_conformer_batch_is_separate() {
    const std::vector<native::Coordinate> coordinates = {
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{1.0, 1.0, 0.0}},
    };
    const native::ConformerBatchView batch{
        2,
        2,
        native::ArrayView<native::Coordinate>(coordinates),
    };
    batch.validate();
    assert(batch.at(1, 0)[1] == 1.0);

    const native::ConformerBatchView invalid{
        3,
        2,
        native::ArrayView<native::Coordinate>(coordinates),
    };
    assert_invalid([&]() { invalid.validate(); });
}


void test_level1_views_are_optional_and_zero_copy() {
    native::OwnedMoleculeData molecule;
    molecule.atom_count = 2;
    molecule.fields = fields({
        native::MoleculeField::STEREO_RECORDS,
        native::MoleculeField::ATOM_TRANSPORT,
        native::MoleculeField::BOND_TRANSPORT,
    });

    auto& stereo = molecule.stereo_records;
    stereo.anchor_kinds = {
        static_cast<std::uint32_t>(native::StereoAnchorKind::ATOM),
    };
    stereo.anchor_offsets = {0};
    stereo.anchor_counts = {1};
    stereo.anchor_indices = {0};
    stereo.carrier_offsets = {0};
    stereo.carrier_counts = {2};
    stereo.carrier_kinds = {
        static_cast<std::uint32_t>(native::StereoCarrierKind::ATOM),
        static_cast<std::uint32_t>(native::StereoCarrierKind::IMPLICIT_H),
    };
    stereo.carrier_has_atoms = {1, 0};
    stereo.carrier_atoms = {1, 0};
    stereo.group_kinds = {
        static_cast<std::uint32_t>(native::StereoGroupKind::NONE),
    };
    stereo.group_ids = {0};
    stereo.group_present = {0};
    stereo.group_scopes.data = {0};
    stereo.group_scopes.offsets = {0};
    stereo.group_scopes.lengths = {0};
    stereo.group_scopes.present = {0};
    stereo.descriptors.data = {'T', 'H'};
    stereo.descriptors.offsets = {0};
    stereo.descriptors.lengths = {2};
    stereo.descriptors.present = {1};
    stereo.source_namespaces.data = {'r', 'd', 'k', 'i', 't'};
    stereo.source_namespaces.offsets = {0};
    stereo.source_namespaces.lengths = {5};
    stereo.source_namespaces.present = {1};
    stereo.source_codes.data = {'C', 'W'};
    stereo.source_codes.offsets = {0};
    stereo.source_codes.lengths = {2};
    stereo.source_codes.present = {1};

    auto& atoms = molecule.atom_transport;
    atoms.atoms = {0};
    atoms.presence_masks = {
        static_cast<std::uint32_t>(native::AtomTransportField::ISOTOPE),
    };
    atoms.isotopes = {13};
    atoms.radical_electrons = {0};
    atoms.has_current_hydrogens = {1};
    atoms.current_hydrogen_presence_masks = {
        static_cast<std::uint32_t>(
            native::HydrogenTransportField::NO_IMPLICIT
        ),
    };
    atoms.current_implicit_hydrogens = {0};
    atoms.current_bracket_explicit_hydrogens = {0};
    atoms.current_no_implicit = {1};
    atoms.has_source_hydrogens = {0};
    atoms.source_hydrogen_presence_masks = {0};
    atoms.source_implicit_hydrogens = {0};
    atoms.source_bracket_explicit_hydrogens = {0};
    atoms.source_no_implicit = {0};
    atoms.source_namespaces.data = {0};
    atoms.source_namespaces.offsets = {0};
    atoms.source_namespaces.lengths = {0};
    atoms.source_namespaces.present = {0};

    auto& bonds = molecule.bond_transport;
    bonds.atom_indices = {{{0, 1}}};
    bonds.display_directions.data = {0};
    bonds.display_directions.offsets = {0};
    bonds.display_directions.lengths = {0};
    bonds.display_directions.present = {1};

    molecule.validate();
    const auto view = molecule.view();
    assert(view.stereo_records.anchor_kinds.data()
           == stereo.anchor_kinds.data());
    assert(view.stereo_records.source_codes.data.data()
           == stereo.source_codes.data.data());
    assert(view.atom_transport.atoms.data() == atoms.atoms.data());
    assert(view.bond_transport.atom_indices.data()
           == bonds.atom_indices.data());

    stereo.anchor_kinds[0] =
        static_cast<std::uint32_t>(native::StereoAnchorKind::AXIS);
    stereo.anchor_counts[0] = 3;
    stereo.anchor_indices = {0, 1, 0};
    molecule.validate();

    for (const std::uint64_t invalid_count : {0u, 1u, 2u}) {
        stereo.anchor_counts[0] = invalid_count;
        assert_invalid([&]() { molecule.validate(); });
    }
    stereo.anchor_counts[0] = 3;

    stereo.anchor_indices[2] = 2;
    assert_invalid([&]() { molecule.validate(); });
    stereo.anchor_indices[2] = 0;

    stereo.anchor_kinds[0] = 4;
    assert_invalid([&]() { molecule.validate(); });
    stereo.anchor_kinds[0] =
        static_cast<std::uint32_t>(native::StereoAnchorKind::AXIS);

    molecule.fields &= ~native::molecule_field_mask(
        native::MoleculeField::ATOM_TRANSPORT
    );
    assert_invalid([&]() { molecule.validate(); });
}


int main() {
    test_array_view();
    test_field_presence_and_generic_validation();
    test_empty_present_field_differs_from_absence();
    test_conformer_batch_is_separate();
    test_level1_views_are_optional_and_zero_copy();
}
