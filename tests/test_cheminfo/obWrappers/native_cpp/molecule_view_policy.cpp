#include "molecule_data.hpp"
#include "openbabel_adapter.hpp"

#include <openbabel/atom.h>
#include <openbabel/bond.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace native = hotpot::native;
namespace obw = hotpot::obwrappers;


static_assert(std::is_base_of_v<native::OwnedMoleculeData, obw::MoleculeData>);
static_assert(!std::is_aggregate_v<obw::MoleculeData>);
static_assert(std::is_constructible_v<
    obw::MoleculeData,
    std::int32_t,
    std::vector<std::int32_t>,
    std::vector<std::int32_t>,
    std::vector<double>,
    std::vector<obw::Coordinate>,
    std::vector<std::uint8_t>,
    std::vector<std::array<std::int32_t, 2>>,
    std::vector<double>,
    std::vector<obw::BondKind>,
    std::vector<std::uint8_t>,
    std::optional<std::array<double, 6>>
>);
static_assert(std::is_same_v<
    decltype(std::declval<const obw::MoleculeData&>().atom_count()),
    std::size_t
>);
static_assert(std::is_same_v<
    decltype(std::declval<const obw::MoleculeData&>().view()),
    native::MoleculeBufferView
>);


obw::MoleculeData molecule_data() {
    return obw::MoleculeData{
        1,
        {6, 7},
        {0, 1},
        {-0.25, 0.25},
        {{{0.0, 0.0, 0.0}}, {{1.4, 0.0, 0.0}}},
        {0, 0},
        {{{0, 1}}},
        {1.0},
        {obw::BondKind::DATIVE},
        {0},
        std::nullopt,
    };
}


std::string compatibility_error(const obw::MoleculeData& molecule) {
    try {
        obw::validate_openbabel_compatibility(molecule.view());
    } catch (const std::invalid_argument& error) {
        return error.what();
    }
    return {};
}


void test_zero_copy_view_and_consumer_policy_split() {
    static_assert(std::is_same_v<obw::BondKind, native::BondKind>);
    const auto molecule = molecule_data();
    const auto view = molecule.view();

    assert(view.atomic_numbers.data() == molecule.atomic_numbers.data());
    assert(view.coordinates.data() == molecule.coordinates.data());
    assert(view.bond_kinds.data() == molecule.bond_kinds.data());
    assert(view.atom_count == molecule.atom_count());
    assert(view.bond_count == molecule.bond_count());

    molecule.validate();
    native::validate_molecule_view(view);
    assert(compatibility_error(molecule)
        == "Open Babel cannot represent dative bonds losslessly");
}


void test_openbabel_keeps_legacy_per_bond_error_priority() {
    auto molecule = molecule_data();
    molecule.atomic_numbers.push_back(8);
    molecule.formal_charges.push_back(0);
    molecule.partial_charges.push_back(0.0);
    molecule.coordinates.push_back({{2.8, 0.0, 0.0}});
    molecule.atom_aromatic.push_back(0);
    molecule.bond_indices.push_back({{1, 7}});
    molecule.bond_orders.push_back(1.0);
    molecule.bond_kinds.push_back(obw::BondKind::SINGLE);
    molecule.bond_aromatic.push_back(0);

    assert(compatibility_error(molecule)
        == "Open Babel cannot represent dative bonds losslessly");
}


void test_non_dative_adapter_result_is_unchanged() {
    auto molecule = molecule_data();
    molecule.bond_kinds[0] = obw::BondKind::SINGLE;

    auto converted = obw::make_obmol(molecule.view());

    assert(converted.NumAtoms() == 2);
    assert(converted.NumBonds() == 1);
    assert(converted.GetAtom(1)->GetAtomicNum() == 6);
    assert(converted.GetAtom(2)->GetAtomicNum() == 7);
    assert(converted.GetAtom(2)->GetFormalCharge() == 1);
    assert(converted.GetAtom(1)->GetX() == 0.0);
    assert(converted.GetAtom(2)->GetX() == 1.4);
    assert(converted.GetBond(0)->GetBondOrder() == 1);
    assert(!converted.GetBond(0)->IsAromatic());
}


void test_compatibility_view_tracks_legacy_vector_mutation() {
    auto molecule = molecule_data();
    molecule.bond_kinds[0] = obw::BondKind::SINGLE;
    molecule.atomic_numbers.push_back(8);
    molecule.formal_charges.push_back(0);
    molecule.partial_charges.push_back(0.0);
    molecule.coordinates.push_back({{2.8, 0.0, 0.0}});
    molecule.atom_aromatic.push_back(0);
    molecule.bond_indices.push_back({{1, 2}});
    molecule.bond_orders.push_back(2.0);
    molecule.bond_kinds.push_back(obw::BondKind::DOUBLE);
    molecule.bond_aromatic.push_back(0);

    const auto view = molecule.view();
    assert(molecule.atom_count() == 3);
    assert(molecule.bond_count() == 2);
    assert(view.atom_count == 3);
    assert(view.bond_count == 2);
    assert(view.atomic_numbers.data() == molecule.atomic_numbers.data());
    assert(view.bond_indices.data() == molecule.bond_indices.data());
    molecule.validate();

    auto converted = obw::make_obmol(view);
    assert(converted.NumAtoms() == 3);
    assert(converted.NumBonds() == 2);
    assert(converted.GetAtom(3)->GetAtomicNum() == 8);
    assert(converted.GetBond(1)->GetBondOrder() == 2);
}


int main() {
    test_zero_copy_view_and_consumer_policy_split();
    test_openbabel_keeps_legacy_per_bond_error_priority();
    test_non_dative_adapter_result_is_unchanged();
    test_compatibility_view_tracks_legacy_vector_mutation();
}
