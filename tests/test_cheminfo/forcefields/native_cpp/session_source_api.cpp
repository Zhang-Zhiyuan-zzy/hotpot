#include "contracts.hpp"
#include "stage_contracts.hpp"
#include "structure_session.hpp"
#include "trajectory.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace ff = hotpot::forcefields;


namespace {


ff::ComplexSessionInput complex_input() {
    return ff::ComplexSessionInput{
        1,
        {63, 7, 6},
        {3, 0, 0},
        {3.0, -1.0, 0.0},
        {{0.0, 0.0, 0.0}, {2.4, 0.0, 0.0}, {3.8, 0.0, 0.0}},
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


void test_session_factories_and_state() {
    static_assert(!std::is_copy_constructible_v<ff::StructureSession>);
    static_assert(!std::is_copy_assignable_v<ff::StructureSession>);
    static_assert(!std::is_move_constructible_v<ff::StructureSession>);
    static_assert(!std::is_move_assignable_v<ff::StructureSession>);

    const ff::ComplexSessionInput input = complex_input();
    input.validate();

    auto coordination = ff::create_coordination_session(input);
    ff::StructureSnapshot snapshot = ff::snapshot_structure(*coordination);
    assert(snapshot.coordinates == input.coordinates);
    assert(snapshot.active_ligand_bond_mask == std::vector<std::uint8_t>({1}));
    assert(snapshot.active_coordination_mask == std::vector<std::uint8_t>({0}));
    assert(snapshot.component_ids == std::vector<std::int32_t>({0, 1, 1}));
    assert(snapshot.ligand_bond_count == 1);
    assert(snapshot.active_bond_count == 1);
    assert(snapshot.coordinate_revision == 0);
    assert(snapshot.topology_revision == 0);

    ff::set_coordination_active_mask(*coordination, {1});
    snapshot = ff::snapshot_structure(*coordination);
    assert(snapshot.active_coordination_mask == std::vector<std::uint8_t>({1}));
    assert(snapshot.active_bond_count == 2);
    assert(snapshot.topology_revision == 1);
    ff::set_coordination_active_mask(*coordination, {1});
    assert(ff::snapshot_structure(*coordination).topology_revision == 1);

    ff::set_ligand_bond_active_mask(*coordination, {0});
    snapshot = ff::snapshot_structure(*coordination);
    assert(snapshot.active_ligand_bond_mask == std::vector<std::uint8_t>({0}));
    assert(snapshot.active_bond_count == 1);
    assert(snapshot.topology_revision == 2);
    ff::set_ligand_bond_active_mask(*coordination, {1});
    snapshot = ff::snapshot_structure(*coordination);
    assert(snapshot.active_bond_count == 2);
    assert(snapshot.topology_revision == 3);

    auto coordinates = input.coordinates;
    coordinates[0] = {0.25, 0.5, 0.75};
    ff::update_structure_coordinates(*coordination, coordinates);
    snapshot = ff::snapshot_structure(*coordination);
    assert(snapshot.coordinates == coordinates);
    assert(snapshot.coordinate_revision == 1);
    ff::update_structure_coordinates(*coordination, coordinates);
    assert(ff::snapshot_structure(*coordination).coordinate_revision == 1);

    auto optimization = ff::create_optimization_session(input);
    snapshot = ff::snapshot_structure(*optimization);
    assert(snapshot.active_coordination_mask == std::vector<std::uint8_t>({1}));
    assert(snapshot.active_bond_count == 2);
    assert(snapshot.topology_revision == 0);
}


void test_contract_validation_and_trajectory() {
    ff::ComplexSessionInput malformed = complex_input();
    malformed.intended_coordination_bonds.push_back({0, 1});
    malformed.intended_coordination_orders.push_back(1.0);
    malformed.intended_coordination_kinds.push_back(ff::BondKind::SINGLE);
    bool rejected_duplicate = false;
    try {
        static_cast<void>(ff::create_coordination_session(malformed));
    } catch (const std::invalid_argument&) {
        rejected_duplicate = true;
    }
    assert(rejected_duplicate);

    ff::NativeTrajectoryBatch trajectory{
        3,
        1,
        1,
        ff::NativeTrajectoryStart::COORDINATION_RESTORATION,
        {},
        {},
        -1,
        -1,
    };
    const auto intact_topology = trajectory.append_topology_revision({{1}, {0}});
    const auto open_topology = trajectory.append_topology_revision({{0}, {0}});
    assert(intact_topology == 0);
    assert(open_topology == 1);
    assert(trajectory.append_topology_revision({{1}, {0}}) == 0);
    trajectory.append(ff::NativeTrajectoryFrame{
        complex_input().coordinates,
        ff::NativeTrajectoryStage::COORDINATION_RESTORATION,
        ff::NativeTrajectoryEvent::COORDINATION_READY,
        std::nullopt,
        0,
        std::nullopt,
        std::nullopt,
        ff::NativeCoordinationFrameEvidence{
            ff::BondIndex{0, 1},
            false,
            1,
            false,
            1,
            0,
            0,
            0,
            std::nullopt,
            1,
            {},
            0.75,
            0.2,
        },
        static_cast<std::int32_t>(open_topology),
    });
    trajectory.select(0);
    trajectory.set_terminal(0);
    trajectory.validate();
    assert(trajectory.frame_count() == 1);

    ff::CoordinationStageResult coordination{
        ff::NativeStageStatus::PARTIAL,
        complex_input().coordinates,
        complex_input().coordinates,
        {0},
        20,
        3,
        1,
        {0},
        {},
        {},
        1,
        0,
        0,
        {"coordination_partial"},
        trajectory,
    };
    coordination.validate();

    ff::ComplexOptimizationResult optimization;
    optimization.selected_coordinates = complex_input().coordinates;
    optimization.terminal_coordinates = complex_input().coordinates;
    optimization.final_active_coordination_mask = {0};
    optimization.untangling_attempt_limit = 30;
    optimization.initial_piercing_count = 0;
    optimization.final_piercing_count = 0;
    optimization.minimum_piercing_count = 0;
    optimization.trajectory = trajectory;
    optimization.validate();

    ff::ComplexWorkflowResult workflow{
        coordination,
        optimization,
        complex_input().coordinates,
        complex_input().coordinates,
        {0},
        {},
        trajectory,
    };
    workflow.validate();
}


}  // namespace


int main() {
    test_session_factories_and_state();
    test_contract_validation_and_trajectory();
}
