#include "optimization_stage.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>


namespace ff = hotpot::forcefields;


namespace hotpot::forcefields::detail {


bool post_checkpoint_requires_topology_blocked_tail(
    std::size_t piercing_pair_count
) noexcept;

bool post_repair_requires_stabilization(
    std::size_t piercing_pair_count,
    bool coordinates_changed
) noexcept;


}  // namespace hotpot::forcefields::detail


namespace {


ff::ComplexSessionInput linear_complex_input() {
    const std::vector<ff::BondIndex> ligand_bonds = {{1, 2}};
    return ff::ComplexSessionInput{
        1,
        {63, 7, 6},
        {0, 0, 0},
        {0.0, 0.0, 0.0},
        {{0.0, 0.0, 0.0}, {2.35, 0.0, 0.0}, {3.75, 0.0, 0.0}},
        {0, 0, 0},
        ligand_bonds,
        {1.0},
        {ff::BondKind::SINGLE},
        {0},
        {0},
        {{0, 1}},
        {1.0},
        {ff::BondKind::DATIVE},
        std::nullopt,
    };
}


ff::ComplexSessionInput piercing_input(double ring_bond_order = 1.5) {
    const std::vector<ff::BondIndex> ligand_bonds = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5},
    };
    return ff::ComplexSessionInput{
        1,
        {6, 6, 6, 6, 6, 6, 63},
        std::vector<std::int32_t>(7, 0),
        std::vector<double>(7, 0.0),
        {
            {-2.0, -2.0, 0.0},
            {2.0, -2.0, 0.0},
            {2.0, 2.0, 0.0},
            {-2.0, 2.0, 0.0},
            {0.0, 0.0, -2.0},
            {0.0, 0.0, 2.0},
            {20.0, 20.0, 20.0},
        },
        std::vector<std::uint8_t>(7, 0),
        ligand_bonds,
        {
            ring_bond_order,
            ring_bond_order,
            ring_bond_order,
            ring_bond_order,
            1.0,
        },
        std::vector<ff::BondKind>(ligand_bonds.size(), ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(ligand_bonds.size(), 0),
        {6},
        {},
        {},
        {},
        std::nullopt,
    };
}


ff::ComplexOptimizationOptions options() {
    ff::ComplexOptimizationOptions value;
    value.epochs = 2;
    value.steps_per_epoch = 1;
    value.untangling_attempt_limit = 2;
    value.retain_epoch_history = true;
    return value;
}


ff::PerturbationOffsetBatch offsets(
    std::size_t atom_count,
    std::size_t frame_count
) {
    return ff::PerturbationOffsetBatch{
        atom_count,
        std::vector<std::vector<ff::Coordinate>>(
            frame_count,
            std::vector<ff::Coordinate>(atom_count, {0.0, 0.0, 0.0})
        ),
    };
}


ff::PerturbationOffsetBatch resolving_offsets() {
    auto batch = offsets(7, 2);
    for (auto& frame : batch.frames) {
        frame[4][0] = 6.0;
        frame[5][0] = 6.0;
    }
    return batch;
}


void test_completed_stage_runs_one_optimizer_without_epoch_scans() {
    auto session = ff::create_optimization_session(linear_complex_input());
    auto stage_options = options();
    stage_options.perturb_interval = 1;
    stage_options.frame_detail = ff::FrameDetail::OPTIMIZATION;
    const auto result = ff::optimize_complex(
        *session,
        stage_options,
        offsets(3, 2),
        offsets(3, 1)
    );
    assert(result.status == ff::NativeStageStatus::COMPLETED);
    assert(result.initial_piercing_count == 0);
    assert(result.final_piercing_count == 0);
    assert(result.untangling_attempts_completed == 0);
    assert(result.epochs_completed == 2);
    assert(result.final_checkpoint.piercing_pair_count == 0);
    assert(result.selected_frame_index
        == result.trajectory.selected_frame_index);
    assert(result.trajectory.selected_frame_index >= 0);
    assert(result.trajectory.terminal_frame_index >= 0);
    assert(std::any_of(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event
                == ff::NativeTrajectoryEvent::EPOCH_COMPLETE;
        }
    ));
    assert(std::count_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event
                == ff::NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT;
        }
    ) == 2);
    assert(std::count_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event
                == ff::NativeTrajectoryEvent::EPOCH_COMPLETE;
        }
    ) == 2);
    const auto& selected = result.trajectory.frames[
        static_cast<std::size_t>(result.selected_frame_index)
    ];
    assert(selected.event == ff::NativeTrajectoryEvent::EPOCH_COMPLETE);
    assert(selected.coordinates == result.selected_coordinates);
    result.validate();
}


void test_minimal_frame_detail_retains_selected_and_terminal_only() {
    auto session = ff::create_optimization_session(linear_complex_input());
    auto stage_options = options();
    stage_options.epochs = 1;
    stage_options.retain_epoch_history = false;
    stage_options.frame_detail = ff::FrameDetail::NONE;
    const auto result = ff::optimize_complex(
        *session,
        stage_options,
        offsets(3, 2),
        offsets(3, 0)
    );
    assert(result.epoch_energies.empty());
    assert(result.epochs_completed == 1);
    assert(std::none_of(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event
                == ff::NativeTrajectoryEvent::EPOCH_COMPLETE;
        }
    ));
    assert(result.selected_frame_index >= 0);
    assert(result.trajectory.terminal_frame_index >= 0);
    assert(result.trajectory.frames[
        static_cast<std::size_t>(result.selected_frame_index)
    ].event == ff::NativeTrajectoryEvent::OPTIMIZED);
    result.validate();
}


void test_post_checkpoint_decisions_do_not_depend_on_coordinate_change() {
    assert(!ff::detail::post_checkpoint_requires_topology_blocked_tail(0));
    assert(ff::detail::post_checkpoint_requires_topology_blocked_tail(1));
    assert(!ff::detail::post_repair_requires_stabilization(1, true));
    assert(!ff::detail::post_repair_requires_stabilization(0, false));
    assert(ff::detail::post_repair_requires_stabilization(0, true));
}


void test_initial_unrepairable_piercing_is_topology_blocked() {
    auto session = ff::create_optimization_session(
        piercing_input()
    );
    const auto initial = ff::snapshot_structure(*session);
    const auto result = ff::optimize_complex(
        *session,
        options(),
        offsets(7, 2),
        offsets(7, 0)
    );
    assert(result.status == ff::NativeStageStatus::PARTIAL);
    assert(!result.untangling_resolved);
    assert(result.initial_piercing_count == 1);
    assert(result.final_piercing_count == 1);
    assert(result.final_checkpoint.piercing_pair_count == 1);
    assert(result.untangling_attempts_completed == 0);
    assert(result.epochs_completed == 0);
    assert(result.steps_submitted == 0);
    assert(result.initialization_steps == 0);
    assert(result.best_epoch == -1);
    assert(std::isnan(result.final_energy_kj_mol));
    assert(std::isnan(result.best_energy_kj_mol));
    assert(std::isnan(result.rms_gradient_kj_mol_angstrom));
    assert(std::isnan(result.max_gradient_kj_mol_angstrom));
    assert(result.termination_reason == "topology_blocked");
    assert(result.selected_frame_index
        == result.trajectory.selected_frame_index);
    const auto& selected = result.trajectory.frames[
        static_cast<std::size_t>(result.selected_frame_index)
    ];
    assert(selected.event
        == ff::NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT);
    assert(std::holds_alternative<ff::NativeRingFrameEvidence>(
        selected.evidence
    ));
    result.validate();
    assert(ff::snapshot_structure(*session).coordinates
        == initial.coordinates);
}


void test_initial_piercing_is_repaired_before_numerical_optimization() {
    auto session = ff::create_optimization_session(piercing_input(1.0));
    auto stage_options = options();
    stage_options.epochs = 1;
    stage_options.frame_detail = ff::FrameDetail::ALL_ATTEMPTS;
    const auto result = ff::optimize_complex(
        *session,
        stage_options,
        resolving_offsets(),
        offsets(7, 0)
    );
    assert(result.initial_piercing_count == 1);
    assert(result.untangling_attempts_completed >= 1);
    assert(result.untangling_resolved);
    assert(result.final_piercing_count == 0);
    assert(result.epochs_completed >= 1);
    assert(std::any_of(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::RING_OPENED;
        }
    ));
    assert(std::count_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event
                == ff::NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT;
        }
    ) == 3);
    const auto opened = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::RING_OPENED;
        }
    );
    const auto perturbed = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::PERTURBED;
        }
    );
    const auto closed = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::RING_CLOSED;
        }
    );
    assert(opened < perturbed && perturbed < closed);
    const auto final_snapshot = ff::snapshot_structure(*session);
    assert(std::all_of(
        final_snapshot.active_ligand_bond_mask.begin(),
        final_snapshot.active_ligand_bond_mask.end(),
        [](std::uint8_t active) { return active != 0; }
    ));
    result.validate();
}


void test_input_streams_are_validated_before_session_mutation() {
    auto session = ff::create_optimization_session(linear_complex_input());
    auto invalid_options = options();
    invalid_options.perturb_interval = 1;
    const auto before = ff::snapshot_structure(*session);
    bool rejected = false;
    try {
        static_cast<void>(ff::optimize_complex(
            *session,
            invalid_options,
            offsets(3, 2),
            offsets(3, 0)
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    const auto after = ff::snapshot_structure(*session);
    assert(after.coordinates == before.coordinates);
    assert(after.active_ligand_bond_mask == before.active_ligand_bond_mask);
    assert(after.active_coordination_mask == before.active_coordination_mask);
    assert(after.coordinate_revision == before.coordinate_revision);
    assert(after.topology_revision == before.topology_revision);
}


void test_incomplete_topology_is_rejected_before_optimization() {
    auto session = ff::create_coordination_session(linear_complex_input());
    const auto before = ff::snapshot_structure(*session);
    bool rejected = false;
    try {
        static_cast<void>(ff::optimize_complex(
            *session,
            options(),
            offsets(3, 2),
            offsets(3, 0)
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    const auto after = ff::snapshot_structure(*session);
    assert(after.coordinates == before.coordinates);
    assert(after.active_coordination_mask
        == before.active_coordination_mask);
}


}  // namespace


int main() {
    test_completed_stage_runs_one_optimizer_without_epoch_scans();
    test_minimal_frame_detail_retains_selected_and_terminal_only();
    test_post_checkpoint_decisions_do_not_depend_on_coordinate_change();
    test_initial_unrepairable_piercing_is_topology_blocked();
    test_initial_piercing_is_repaired_before_numerical_optimization();
    test_input_streams_are_validated_before_session_mutation();
    test_incomplete_topology_is_rejected_before_optimization();
}
