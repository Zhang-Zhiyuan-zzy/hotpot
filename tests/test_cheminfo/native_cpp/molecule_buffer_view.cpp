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


int main() {
    test_array_view();
    test_field_presence_and_generic_validation();
    test_empty_present_field_differs_from_absence();
    test_conformer_batch_is_separate();
}
