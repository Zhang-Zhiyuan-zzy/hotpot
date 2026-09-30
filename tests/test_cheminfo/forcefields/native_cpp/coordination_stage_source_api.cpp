#include "coordination_stage.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <optional>
#include <vector>


namespace ff = hotpot::forcefields;


namespace {


ff::ComplexSessionInput make_input(
    std::vector<std::int32_t> atomic_numbers,
    std::vector<ff::Coordinate> coordinates,
    std::vector<ff::BondIndex> ligand_bonds,
    std::vector<ff::BondIndex> coordination_bonds
) {
    const std::size_t atom_count = coordinates.size();
    return ff::ComplexSessionInput{
        1,
        std::move(atomic_numbers),
        std::vector<std::int32_t>(atom_count, 0),
        std::vector<double>(atom_count, 0.0),
        std::move(coordinates),
        std::vector<std::uint8_t>(atom_count, 0),
        ligand_bonds,
        std::vector<double>(ligand_bonds.size(), 1.0),
        std::vector<ff::BondKind>(ligand_bonds.size(), ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(ligand_bonds.size(), 0),
        {0},
        coordination_bonds,
        std::vector<double>(coordination_bonds.size(), 1.0),
        std::vector<ff::BondKind>(
            coordination_bonds.size(),
            ff::BondKind::DATIVE
        ),
        std::nullopt,
    };
}


ff::CoordinationStageOptions one_attempt_options() {
    ff::CoordinationStageOptions options;
    options.attempt_limit = 1;
    options.relaxation_steps = 1;
    options.frame_detail = ff::FrameDetail::ALL_ATTEMPTS;
    return options;
}


void test_empty_coordination_stage_is_a_selected_terminal_state() {
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6},
        {{0.0, 0.0, 0.0}, {2.4, 0.0, 0.0}, {3.8, 0.0, 0.0}},
        {{1, 2}},
        {}
    ));
    const ff::PerturbationOffsetBatch offsets{3, {}};
    const auto result = ff::restore_coordination(
        *session,
        one_attempt_options(),
        offsets
    );
    assert(result.status == ff::NativeStageStatus::COMPLETED);
    assert(result.bond_count == 0);
    assert(result.final_active_coordination_mask.empty());
    assert(result.placement_report.metals.empty());
    assert(result.trajectory.frame_count() == 2);
    assert(result.trajectory.selected_frame_index == 1);
    assert(result.trajectory.terminal_frame_index == 1);
}


void test_safe_bond_is_restored_and_relaxed() {
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6},
        {{0.0, 0.0, 0.0}, {2.35, 0.0, 0.0}, {3.75, 0.0, 0.0}},
        {{1, 2}},
        {{0, 1}}
    ));
    const ff::PerturbationOffsetBatch offsets{3, {}};
    const auto result = ff::restore_coordination(
        *session,
        one_attempt_options(),
        offsets
    );
    assert(result.status == ff::NativeStageStatus::COMPLETED);
    assert(result.bond_count == 1);
    assert(result.attempts_completed == 0);
    assert(result.final_active_coordination_mask
        == std::vector<std::uint8_t>({1}));
    assert(result.forced_bond_keys.empty());
    assert(result.placement_report.metals.size() == 1);
    assert(result.trajectory.selected_frame_index
        == result.trajectory.terminal_frame_index);
    assert(std::any_of(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::BOND_ACCEPTED;
        }
    ));
    const auto trial = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::BOND_TRIAL;
        }
    );
    const auto accepted = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::BOND_ACCEPTED;
        }
    );
    const auto optimized = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::OPTIMIZED;
        }
    );
    assert(trial < accepted);
    assert(accepted < optimized);
    assert(std::none_of(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::BOND_ROLLBACK;
        }
    ));
    assert(ff::snapshot_structure(*session).active_coordination_mask
        == std::vector<std::uint8_t>({1}));
}


void test_piercing_bond_is_forced_without_post_force_optimization() {
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6, 6, 6, 6},
        {
            {0.0, 0.0, 1.34},
            {0.0, 0.0, -1.34},
            {-3.0, -3.0, 0.0},
            {3.0, -3.0, 0.0},
            {3.0, 3.0, 0.0},
            {-3.0, 3.0, 0.0},
        },
        {{2, 3}, {3, 4}, {4, 5}, {5, 2}},
        {{0, 1}}
    ));
    auto options = one_attempt_options();
    options.placement.maximum_candidate_count = 1;
    const auto result = ff::restore_coordination(
        *session,
        options,
        ff::PerturbationOffsetBatch{6, {}}
    );
    assert(result.status == ff::NativeStageStatus::PARTIAL);
    assert(result.forced_bond_keys == std::vector<ff::BondIndex>({{0, 1}}));
    assert(result.final_active_coordination_mask
        == std::vector<std::uint8_t>({1}));
    const auto forced = std::find_if(
        result.trajectory.frames.begin(),
        result.trajectory.frames.end(),
        [](const ff::NativeTrajectoryFrame& frame) {
            return frame.event == ff::NativeTrajectoryEvent::BOND_FORCED;
        }
    );
    assert(forced != result.trajectory.frames.end());
    assert(std::next(forced)->event == ff::NativeTrajectoryEvent::TERMINAL);
}


void test_offset_schedule_is_explicit_and_exact() {
    auto session = ff::create_coordination_session(make_input(
        {63, 7, 6},
        {{0.0, 0.0, 0.0}, {2.35, 0.0, 0.0}, {3.75, 0.0, 0.0}},
        {{1, 2}},
        {{0, 1}}
    ));
    auto options = one_attempt_options();
    options.attempt_limit = 2;
    bool rejected = false;
    try {
        static_cast<void>(ff::restore_coordination(
            *session,
            options,
            ff::PerturbationOffsetBatch{3, {}}
        ));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}


}  // namespace


int main() {
    test_empty_coordination_stage_is_a_selected_terminal_state();
    test_safe_bond_is_restored_and_relaxed();
    test_piercing_bond_is_forced_without_post_force_optimization();
    test_offset_schedule_is_explicit_and_exact();
}
