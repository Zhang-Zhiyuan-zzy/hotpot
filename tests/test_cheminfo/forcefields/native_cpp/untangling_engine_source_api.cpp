#include "untangling_engine.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>


namespace ff = hotpot::forcefields;
namespace detail = hotpot::forcefields::detail;


namespace {


ff::ComplexSessionInput square_piercing_input(double ring_bond_order = 1.0) {
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
        {
            ff::BondKind::SINGLE,
            ff::BondKind::SINGLE,
            ff::BondKind::SINGLE,
            ff::BondKind::SINGLE,
            ff::BondKind::SINGLE,
        },
        std::vector<std::uint8_t>(ligand_bonds.size(), 0),
        {6},
        {},
        {},
        {},
        std::nullopt,
    };
}


ff::ComplexSessionInput metal_ring_piercing_input() {
    const std::vector<ff::BondIndex> ligand_bonds = {
        {1, 2}, {2, 3}, {4, 5},
    };
    const std::vector<ff::BondIndex> coordination_bonds = {
        {0, 1}, {0, 3},
    };
    return ff::ComplexSessionInput{
        1,
        {63, 6, 6, 6, 6, 6},
        std::vector<std::int32_t>(6, 0),
        std::vector<double>(6, 0.0),
        {
            {-2.0, -2.0, 0.0},
            {2.0, -2.0, 0.0},
            {2.0, 2.0, 0.0},
            {-2.0, 2.0, 0.0},
            {0.0, 0.0, -2.0},
            {0.0, 0.0, 2.0},
        },
        std::vector<std::uint8_t>(6, 0),
        ligand_bonds,
        std::vector<double>(ligand_bonds.size(), 1.0),
        std::vector<ff::BondKind>(ligand_bonds.size(), ff::BondKind::SINGLE),
        std::vector<std::uint8_t>(ligand_bonds.size(), 0),
        {0},
        coordination_bonds,
        std::vector<double>(coordination_bonds.size(), 1.0),
        std::vector<ff::BondKind>(
            coordination_bonds.size(), ff::BondKind::DATIVE
        ),
        std::nullopt,
    };
}


ff::ComplexSessionInput two_targets_one_ring_input() {
    auto input = square_piercing_input();
    input.atomic_numbers.insert(input.atomic_numbers.end(), {6, 6});
    input.formal_charges.insert(input.formal_charges.end(), {0, 0});
    input.partial_charges.insert(input.partial_charges.end(), {0.0, 0.0});
    input.coordinates.insert(
        input.coordinates.end(),
        {{0.75, 0.0, -2.0}, {0.75, 0.0, 2.0}}
    );
    input.atom_aromatic.insert(input.atom_aromatic.end(), {0, 0});
    input.ligand_bond_indices.push_back({7, 8});
    input.ligand_bond_orders.push_back(1.0);
    input.ligand_bond_kinds.push_back(ff::BondKind::SINGLE);
    input.ligand_bond_aromatic.push_back(0);
    return input;
}


ff::RingUntanglingOptions options() {
    ff::RingUntanglingOptions value;
    value.short_optimization_steps = 1;
    return value;
}


detail::BondRingCheckpoint checkpoint(
    ff::StructureSession& session,
    const ff::RingUntanglingOptions& engine_options
) {
    const auto snapshot = ff::snapshot_structure(session);
    const auto workspace = detail::prepare_ring_workspace(
        session.input(),
        snapshot.coordinates,
        snapshot.active_ligand_bond_mask,
        snapshot.active_coordination_mask,
        engine_options.ring_workspace
    );
    return detail::scan_bond_ring_checkpoint(
        workspace, engine_options.ring_workspace
    );
}


ff::PerturbationOffsetBatch offsets(
    std::size_t frame_count,
    std::size_t atom_count = 7,
    ff::Coordinate offset = {0.01, 0.02, 0.03}
) {
    return ff::PerturbationOffsetBatch{
        atom_count,
        std::vector<std::vector<ff::Coordinate>>(
            frame_count,
            std::vector<ff::Coordinate>(atom_count, offset)
        ),
    };
}


void test_no_piercing_consumes_no_shared_attempt() {
    auto input = square_piercing_input();
    input.coordinates[4] = {4.0, 0.0, -2.0};
    input.coordinates[5] = {4.0, 0.0, 2.0};
    auto session = ff::create_optimization_session(input);
    const auto engine_options = options();
    const auto entry = checkpoint(*session, engine_options);
    assert(entry.piercing_pair_count == 0);
    ff::RingUntanglingAttemptCursor cursor{2, 0};
    const auto result = ff::untangle_ring_piercings(
        *session, engine_options, offsets(2), cursor, entry
    );
    assert(result.resolved);
    assert(result.attempts_used == 0);
    assert(cursor.attempts_completed == 0);
    assert(result.steps.size() == 1);
}


void test_open_attempt_is_observable_and_exactly_restores_topology() {
    auto session = ff::create_optimization_session(square_piercing_input());
    const auto engine_options = options();
    const auto entry = checkpoint(*session, engine_options);
    assert(entry.piercing_pair_count == 1);
    const auto initial = ff::snapshot_structure(*session);
    ff::RingUntanglingAttemptCursor cursor{1, 0};
    const auto result = ff::untangle_ring_piercings(
        *session, engine_options, offsets(1), cursor, entry
    );
    assert(cursor.attempts_completed == 1);
    assert(result.attempts_used == 1);
    const auto opened = std::find_if(
        result.steps.begin(), result.steps.end(),
        [](const ff::RingUntanglingStep& step) {
            return step.event == ff::RingUntanglingEvent::RING_OPENED;
        }
    );
    const auto perturbed = std::find_if(
        result.steps.begin(), result.steps.end(),
        [](const ff::RingUntanglingStep& step) {
            return step.event == ff::RingUntanglingEvent::PERTURBED;
        }
    );
    const auto optimized = std::find_if(
        result.steps.begin(), result.steps.end(),
        [](const ff::RingUntanglingStep& step) {
            return step.event
                == ff::RingUntanglingEvent::OPEN_TOPOLOGY_OPTIMIZED;
        }
    );
    const auto closed = std::find_if(
        result.steps.begin(), result.steps.end(),
        [](const ff::RingUntanglingStep& step) {
            return step.event == ff::RingUntanglingEvent::RING_CLOSED;
        }
    );
    assert(opened < perturbed && perturbed < optimized && optimized < closed);
    assert(opened->opening_bond_key == ff::BondIndex({0, 1}));
    assert(opened->snapshot.active_ligand_bond_mask[0] == 0);
    assert(opened->snapshot.topology_revision
        == initial.topology_revision + 1);
    assert(perturbed->snapshot.coordinates[0][0]
        == opened->snapshot.coordinates[0][0] + 0.01);
    assert(perturbed->snapshot.coordinates[0][1]
        == opened->snapshot.coordinates[0][1] + 0.02);
    assert(perturbed->snapshot.coordinates[0][2]
        == opened->snapshot.coordinates[0][2] + 0.03);
    assert(optimized->energy_kj_mol.has_value());
    assert(!closed->energy_kj_mol.has_value());
    assert(closed->snapshot.active_ligand_bond_mask
        == initial.active_ligand_bond_mask);
    assert(closed->snapshot.topology_revision
        == initial.topology_revision + 2);
    assert(ff::snapshot_structure(*session).active_ligand_bond_mask
        == initial.active_ligand_bond_mask);
}


void test_no_eligible_edge_returns_inspectable_closed_structure() {
    auto session = ff::create_optimization_session(
        square_piercing_input(1.5)
    );
    const auto engine_options = options();
    const auto entry = checkpoint(*session, engine_options);
    ff::RingUntanglingAttemptCursor cursor{2, 0};
    const auto result = ff::untangle_ring_piercings(
        *session, engine_options, offsets(2), cursor, entry
    );
    assert(!result.resolved);
    assert(result.attempts_used == 0);
    assert(cursor.attempts_completed == 0);
    assert(result.full_checkpoint_count == 0);
    assert(std::count_if(
        result.steps.begin(),
        result.steps.end(),
        [](const ff::RingUntanglingStep& step) {
            return step.event
                == ff::RingUntanglingEvent::TOPOLOGY_CHECKPOINT;
        }
    ) == 1);
    assert(result.warning_codes
        == std::vector<std::string>({"ring_piercing_has_no_opening_edge"}));
    const auto final_snapshot = ff::snapshot_structure(*session);
    assert(std::all_of(
        final_snapshot.active_ligand_bond_mask.begin(),
        final_snapshot.active_ligand_bond_mask.end(),
        [](std::uint8_t active) { return active != 0; }
    ));
}


void test_forcefield_setup_failure_restores_opened_bond() {
    auto session = ff::create_optimization_session(square_piercing_input());
    auto engine_options = options();
    engine_options.forcefield = "missing_forcefield";
    const auto entry = checkpoint(*session, engine_options);
    const auto original_mask = ff::snapshot_structure(
        *session
    ).active_ligand_bond_mask;
    ff::RingUntanglingAttemptCursor cursor{1, 0};
    bool failed = false;
    try {
        static_cast<void>(ff::untangle_ring_piercings(
            *session, engine_options, offsets(1), cursor, entry
        ));
    } catch (const std::runtime_error&) {
        failed = true;
    }
    assert(failed);
    assert(cursor.attempts_completed == 0);
    assert(ff::snapshot_structure(*session).active_ligand_bond_mask
        == original_mask);
}


void test_metal_ring_opens_only_an_intended_coordination_edge() {
    auto session = ff::create_optimization_session(
        metal_ring_piercing_input()
    );
    const auto engine_options = options();
    const auto entry = checkpoint(*session, engine_options);
    assert(entry.piercing_pair_count == 1);
    ff::RingUntanglingAttemptCursor cursor{1, 0};
    const auto result = ff::untangle_ring_piercings(
        *session, engine_options, offsets(1, 6), cursor, entry
    );
    const auto opened = std::find_if(
        result.steps.begin(), result.steps.end(),
        [](const ff::RingUntanglingStep& step) {
            return step.event == ff::RingUntanglingEvent::RING_OPENED;
        }
    );
    assert(opened != result.steps.end());
    assert(opened->opening_bond_key == ff::BondIndex({0, 1}));
    assert(opened->snapshot.active_ligand_bond_mask
        == std::vector<std::uint8_t>({1, 1, 1}));
    assert(opened->snapshot.active_coordination_mask
        == std::vector<std::uint8_t>({0, 1}));
    assert(ff::snapshot_structure(*session).active_coordination_mask
        == std::vector<std::uint8_t>({1, 1}));
}


void test_attempt_cursor_is_shared_across_calls() {
    auto session = ff::create_optimization_session(square_piercing_input());
    const auto engine_options = options();
    ff::RingUntanglingAttemptCursor cursor{1, 0};
    const auto perturbations = offsets(1);
    static_cast<void>(ff::untangle_ring_piercings(
        *session,
        engine_options,
        perturbations,
        cursor,
        checkpoint(*session, engine_options)
    ));
    assert(cursor.exhausted());
    const auto second = ff::untangle_ring_piercings(
        *session,
        engine_options,
        perturbations,
        cursor,
        checkpoint(*session, engine_options)
    );
    assert(second.attempts_used == 0);
    assert(cursor.attempts_completed == 1);
}


void test_watch_prepares_a_shared_ring_once_for_multiple_targets() {
    auto session = ff::create_optimization_session(
        two_targets_one_ring_input()
    );
    const auto engine_options = options();
    const auto entry = checkpoint(*session, engine_options);
    assert(entry.piercing_pair_count == 2);
    ff::RingUntanglingAttemptCursor cursor{1, 0};
    const auto result = ff::untangle_ring_piercings(
        *session, engine_options, offsets(1, 9), cursor, entry
    );
    assert(result.attempts_used == 1);
    assert(result.watch_cycle_preparation_count == 1);
}


}  // namespace


int main() {
    test_no_piercing_consumes_no_shared_attempt();
    test_open_attempt_is_observable_and_exactly_restores_topology();
    test_no_eligible_edge_returns_inspectable_closed_structure();
    test_forcefield_setup_failure_restores_opened_bond();
    test_metal_ring_opens_only_an_intended_coordination_edge();
    test_attempt_cursor_is_shared_across_calls();
    test_watch_prepares_a_shared_ring_once_for_multiple_targets();
}
