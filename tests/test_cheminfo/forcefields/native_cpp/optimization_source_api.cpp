#include "session_optimization.hpp"

#include "openbabel_adapter.hpp"

#include <openbabel/mol.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>


namespace ff = hotpot::forcefields;
namespace obw = hotpot::obwrappers;


namespace {


obw::MoleculeData molecule_data() {
    return obw::MoleculeData{
        1,
        {6, 6},
        {0, 0},
        {0.0, 0.0},
        {{0.0, 0.0, 0.0}, {2.2, 0.0, 0.0}},
        {0, 0},
        {{{0, 1}}},
        {1.0},
        {obw::BondKind::SINGLE},
        {0},
        std::nullopt,
    };
}


ff::ComplexSessionInput complex_input() {
    return ff::ComplexSessionInput{
        1,
        {63, 7, 6},
        {3, 0, 0},
        {3.0, -1.0, 0.0},
        {{0.0, 0.0, 0.0}, {2.4, 0.0, 0.0}, {4.4, 0.0, 0.0}},
        {0, 0, 0},
        {{{1, 2}}},
        {1.0},
        {ff::BondKind::SINGLE},
        {0},
        {0},
        {{{0, 1}}},
        {1.0},
        {ff::BondKind::DATIVE},
        std::nullopt,
    };
}


bool same_coordinates(
    const std::vector<obw::Coordinate>& left,
    const std::vector<obw::Coordinate>& right
) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t atom = 0; atom < left.size(); ++atom) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (std::abs(left[atom][axis] - right[atom][axis]) > 1.0e-12) {
                return false;
            }
        }
    }
    return true;
}


obw::OptimizationOptions optimization_options() {
    return obw::OptimizationOptions{
        "UFF",
        "steepest",
        2,
        2,
        std::nullopt,
        true,
        true,
        false,
        0.0,
        12.5,
        1.0e-6,
        std::nullopt,
    };
}


void test_direct_obmol_entries_leave_selected_coordinates() {
    const auto input = molecule_data();
    auto molecule = obw::make_obmol(input);
    const auto single = obw::single_optimize_in_place(
        molecule, "UFF", 2, 1.0e-8, 0.05
    );
    assert(same_coordinates(
        obw::extract_coordinates(molecule), single.coordinates
    ));

    const auto iterative = obw::optimize_in_place(
        molecule, optimization_options(), {}, 1.0e-8, 0.05
    );
    assert(same_coordinates(
        obw::extract_coordinates(molecule), iterative.coordinates
    ));
    assert(iterative.terminal_coordinates.size() == input.atom_count());
}


void test_iterative_entry_restores_selected_not_terminal_frame() {
    auto molecule = obw::make_obmol(molecule_data());
    auto options = optimization_options();
    options.epochs = 2;
    options.steps_per_epoch = 1;
    options.perturb_interval = 1;
    const std::vector<std::vector<obw::Coordinate>> offsets{
        {{20.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    };
    const auto result = obw::optimize_in_place(
        molecule, options, offsets, 1.0e-8, 0.05
    );
    assert(!same_coordinates(result.coordinates, result.terminal_coordinates));
    assert(same_coordinates(
        obw::extract_coordinates(molecule), result.coordinates
    ));
}


void test_molecule_data_and_obmol_entries_share_results() {
    const auto input = molecule_data();
    const auto packed = obw::single_optimize(
        input, "UFF", 2, 1.0e-8, 0.05
    );
    auto molecule = obw::make_obmol(input);
    const auto direct = obw::single_optimize_in_place(
        molecule, "UFF", 2, 1.0e-8, 0.05
    );
    assert(same_coordinates(packed.coordinates, direct.coordinates));
    assert(std::abs(packed.energy_kj_mol - direct.energy_kj_mol) < 1.0e-12);
}


void test_in_place_failure_restores_coordinates() {
    auto molecule = obw::make_obmol(molecule_data());
    const auto before = obw::extract_coordinates(molecule);
    bool rejected = false;
    try {
        static_cast<void>(obw::single_optimize_in_place(
            molecule, "not-a-forcefield", 1, 1.0e-8, 0.05
        ));
    } catch (const obw::ForceFieldSetupFailure&) {
        rejected = true;
    }
    assert(rejected);
    assert(same_coordinates(before, obw::extract_coordinates(molecule)));
}


void test_session_adapter_updates_only_coordinate_revision() {
    auto session = ff::create_coordination_session(complex_input());
    const auto before = ff::snapshot_structure(*session);
    const auto result = ff::single_optimize_session(
        *session, "UFF", 2, 1.0e-8, 0.05
    );
    const auto after = ff::snapshot_structure(*session);
    assert(same_coordinates(after.coordinates, result.coordinates));
    assert(after.coordinate_revision == before.coordinate_revision + (
        same_coordinates(before.coordinates, after.coordinates) ? 0 : 1
    ));
    assert(after.topology_revision == before.topology_revision);
    assert(after.active_ligand_bond_mask == before.active_ligand_bond_mask);
    assert(after.active_coordination_mask == before.active_coordination_mask);

    const auto iterative = ff::optimize_session(
        *session,
        optimization_options(),
        ff::PerturbationOffsetBatch{session->atom_count(), {}},
        1.0e-8,
        0.05
    );
    const auto optimized = ff::snapshot_structure(*session);
    assert(same_coordinates(optimized.coordinates, iterative.coordinates));
    assert(optimized.coordinate_revision == after.coordinate_revision + (
        same_coordinates(after.coordinates, optimized.coordinates) ? 0 : 1
    ));
    assert(optimized.topology_revision == after.topology_revision);

    const auto accepted = optimized.coordinates;
    bool rejected = false;
    try {
        static_cast<void>(ff::single_optimize_session(
            *session, "not-a-forcefield", 1, 1.0e-8, 0.05
        ));
    } catch (const obw::ForceFieldSetupFailure&) {
        rejected = true;
    }
    assert(rejected);
    const auto restored = ff::snapshot_structure(*session);
    assert(same_coordinates(restored.coordinates, accepted));
    assert(restored.coordinate_revision == optimized.coordinate_revision);
    assert(restored.topology_revision == optimized.topology_revision);
}


}  // namespace


int main() {
    test_direct_obmol_entries_leave_selected_coordinates();
    test_iterative_entry_restores_selected_not_terminal_frame();
    test_molecule_data_and_obmol_entries_share_results();
    test_in_place_failure_restores_coordinates();
    test_session_adapter_updates_only_coordinate_revision();
}
