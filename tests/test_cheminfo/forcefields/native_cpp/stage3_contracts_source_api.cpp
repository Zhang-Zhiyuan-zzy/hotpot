#include "stage_contracts.hpp"
#include "topology_workspace.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>


namespace ff = hotpot::forcefields;
namespace geo = hotpot::geometry;


namespace {


template <typename Operation>
void expect_invalid(Operation operation) {
    bool rejected = false;
    try {
        operation();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}


std::vector<ff::Coordinate> selected_coordinates() {
    return {
        {0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };
}


std::vector<ff::Coordinate> terminal_coordinates() {
    auto coordinates = selected_coordinates();
    coordinates[3] = {0.0, 0.0, 1.1};
    return coordinates;
}


ff::NativeTrajectoryBatch trajectory(bool select_frame = true) {
    ff::NativeTrajectoryBatch result{
        4,
        1,
        1,
        ff::NativeTrajectoryStart::COMPLEX_UNTANGLING,
        {},
        {},
        -1,
        -1,
    };
    const auto topology = result.append_topology_revision({{1}, {1}});
    result.append({
        selected_coordinates(),
        ff::NativeTrajectoryStage::FINAL_OPTIMIZATION,
        ff::NativeTrajectoryEvent::OPTIMIZED,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        1.0,
        ff::NativeOptimizationFrameEvidence{},
        static_cast<std::int32_t>(topology),
    });
    result.append({
        terminal_coordinates(),
        ff::NativeTrajectoryStage::FINAL_OPTIMIZATION,
        ff::NativeTrajectoryEvent::TERMINAL,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        1.1,
        ff::NativeOptimizationFrameEvidence{},
        static_cast<std::int32_t>(topology),
    });
    if (select_frame) {
        result.select(0);
    }
    result.set_terminal(1);
    return result;
}


ff::NativeRingCheckpointReport clear_checkpoint() {
    return {};
}


ff::NativeRingCheckpointReport piercing_checkpoint() {
    return ff::NativeRingCheckpointReport{
        geo::PiercingState::PIERCES,
        ff::NativeRingGraphScope::FULL_GRAPH,
        16,
        10000,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        1,
        0,
        0,
        true,
        {{
            0,
            {0, 1, 2},
            {2, 3},
            geo::PiercingState::PIERCES,
            {},
            false,
            true,
        }},
    };
}


ff::ComplexOptimizationResult completed_result() {
    return ff::ComplexOptimizationResult{
        ff::NativeStageStatus::COMPLETED,
        selected_coordinates(),
        terminal_coordinates(),
        {1},
        30,
        0,
        0,
        0,
        0,
        true,
        0,
        0,
        1.1,
        1.0,
        0.1,
        0.2,
        {},
        {},
        {},
        false,
        false,
        false,
        1,
        100,
        1,
        1,
        "kJ/mol",
        "budget_exhausted",
        {},
        trajectory(),
        clear_checkpoint(),
    };
}


ff::ComplexOptimizationResult topology_blocked_result() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    auto blocked_trajectory = trajectory();
    return ff::ComplexOptimizationResult{
        ff::NativeStageStatus::PARTIAL,
        selected_coordinates(),
        terminal_coordinates(),
        {1},
        30,
        0,
        1,
        1,
        1,
        false,
        0,
        -1,
        nan,
        nan,
        nan,
        nan,
        {},
        {},
        {},
        false,
        false,
        false,
        0,
        0,
        0,
        0,
        "",
        "topology_blocked",
        {"ring_piercing_blocks_complex_optimization"},
        std::move(blocked_trajectory),
        piercing_checkpoint(),
    };
}


void test_public_options_and_full_graph_conversion() {
    ff::ComplexOptimizationOptions options;
    options.validate();
    assert(options.ring_screening.maximum_actionable_ring_size == 16);
    assert(options.ring_screening.maximum_relevant_cycle_count == 10000);

    const auto workspace = ff::full_graph_ring_workspace_options(
        options.ring_screening
    );
    assert(workspace.scope == ff::detail::RingGraphScope::FULL_GRAPH);
    assert(workspace.maximum_actionable_ring_size == 16);
    assert(workspace.maximum_relevant_cycle_count == 10000);

    auto invalid = options;
    invalid.torsion_singularity_threshold = 1.0;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = options;
    invalid.ring_screening.maximum_actionable_ring_size = 2;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = options;
    invalid.ring_screening.maximum_relevant_cycle_count = 0;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = options;
    invalid.ring_screening.geometry_tolerances.absolute_length = 0.0;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = options;
    invalid.ring_screening.surface_limits.maximum_surface_count = 0;
    expect_invalid([&invalid]() { invalid.validate(); });
}


void test_checkpoint_dto_conversion_preserves_facts() {
    geo::SegmentCycleRelation piercing_relation{};
    piercing_relation.state = geo::PiercingState::PIERCES;
    geo::SegmentCycleRelation uncertain_relation{};
    uncertain_relation.state = geo::PiercingState::UNDETERMINED;
    uncertain_relation.indeterminacy_causes = {
        geo::SegmentCycleIndeterminacy::NUMERIC_BAND,
    };
    ff::detail::BondRingCheckpoint checkpoint{
        geo::PiercingState::PIERCES,
        ff::detail::RingGraphScope::FULL_GRAPH,
        16,
        10000,
        1,
        1,
        0,
        3,
        2,
        0,
        2,
        1,
        0,
        1,
        false,
        {
            {0, {0, 1, 2}, {2, 3}, piercing_relation, false, true},
            {0, {0, 1, 2}, {1, 3}, uncertain_relation, false, false},
        },
    };

    const auto report = ff::native_ring_checkpoint_report(checkpoint);
    assert(report.scope == ff::NativeRingGraphScope::FULL_GRAPH);
    assert(report.piercing_pair_count == 1);
    assert(report.undetermined_pair_count == 1);
    assert(report.actionable_findings.size() == 2);
    assert(report.actionable_findings[0].ring_atom_indices
        == std::vector<std::size_t>({0, 1, 2}));
    assert(report.actionable_findings[0].bond_key
        == ff::BondIndex({2, 3}));
    assert(!report.actionable_findings[0].aabb_separated);
    assert(!report.actionable_findings[1].surface_complete);
    assert(!report.scan_complete);
    assert(report.actionable_findings[1].indeterminacy_causes
        == std::vector<geo::SegmentCycleIndeterminacy>({
            geo::SegmentCycleIndeterminacy::NUMERIC_BAND,
        }));

    checkpoint.scope = ff::detail::RingGraphScope::LIGAND_SKELETON;
    assert(ff::native_ring_checkpoint_report(checkpoint).scope
        == ff::NativeRingGraphScope::LIGAND_SKELETON);
}


void test_complex_result_history_and_cross_field_contracts() {
    auto result = completed_result();
    result.validate();
    result.epoch_energies = {1.0};
    result.validate();

    auto invalid = result;
    invalid.epoch_energies = {1.0, 0.9};
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.selected_frame_index = 1;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.selected_frame_index = -1;
    invalid.trajectory.selected_frame_index = -1;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.trajectory.terminal_frame_index = -1;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.final_piercing_count = 1;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.untangling_resolved = false;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.final_active_coordination_mask = {0};
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.final_checkpoint.scope =
        ff::NativeRingGraphScope::LIGAND_SKELETON;
    expect_invalid([&invalid]() { invalid.validate(); });
}


void test_topology_blocked_contract() {
    auto result = topology_blocked_result();
    result.validate();
    assert(result.selected_frame_index == 0);
    assert(result.trajectory.selected_frame_index == 0);

    auto accumulated = result;
    accumulated.epochs_completed = 2;
    accumulated.steps_submitted = 200;
    accumulated.initialization_steps = 1;
    accumulated.selected_segment_epochs_completed = 2;
    accumulated.energy_changes = {0.2};
    accumulated.max_displacements = {0.1};
    accumulated.epoch_energies = {2.0, 1.0};
    accumulated.validate();

    auto invalid = result;
    invalid.selected_frame_index = -1;
    invalid.trajectory.selected_frame_index = -1;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.trajectory.terminal_frame_index = -1;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.best_epoch = 0;
    expect_invalid([&invalid]() { invalid.validate(); });
    for (const auto member : {
        &ff::ComplexOptimizationResult::final_energy_kj_mol,
        &ff::ComplexOptimizationResult::best_energy_kj_mol,
        &ff::ComplexOptimizationResult::rms_gradient_kj_mol_angstrom,
        &ff::ComplexOptimizationResult::max_gradient_kj_mol_angstrom,
    }) {
        invalid = result;
        invalid.*member = 0.0;
        expect_invalid([&invalid]() { invalid.validate(); });
    }
    invalid = result;
    invalid.converged = true;
    expect_invalid([&invalid]() { invalid.validate(); });
    invalid = result;
    invalid.terminal_converged = true;
    expect_invalid([&invalid]() { invalid.validate(); });
}


}  // namespace


int main() {
    test_public_options_and_full_graph_conversion();
    test_checkpoint_dto_conversion_preserves_facts();
    test_complex_result_history_and_cross_field_contracts();
    test_topology_blocked_contract();
}
