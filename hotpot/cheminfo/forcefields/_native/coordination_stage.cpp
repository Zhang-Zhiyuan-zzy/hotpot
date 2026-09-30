#include "coordination_stage.hpp"

#include "placement_engine.hpp"
#include "session_optimization.hpp"
#include "topology_workspace.hpp"

#include "../../geometry/_native/types.hpp"
#include "../../obWrappers/_native/native_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::forcefields {
namespace {


BondIndex canonical_bond_key(BondIndex endpoints) noexcept {
    if (endpoints[1] < endpoints[0]) {
        std::swap(endpoints[0], endpoints[1]);
    }
    return endpoints;
}


void append_unique(
    std::vector<std::string>& values,
    const std::string& value
) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}


class CoordinationTrajectoryRecorder final {
public:
    CoordinationTrajectoryRecorder(
        StructureSession& session,
        NativeTrajectoryStart start
    ) :
        session_(session),
        enabled_(
            static_cast<std::int32_t>(start)
            <= static_cast<std::int32_t>(
                NativeTrajectoryStart::COORDINATION_RESTORATION
            )
        ),
        batch_{
            session.atom_count(),
            session.ligand_bond_count(),
            session.intended_coordination_bond_count(),
            start,
            {},
            {},
            -1,
            -1,
        } {}

    std::optional<std::size_t> append(
        NativeTrajectoryEvent event,
        std::optional<std::size_t> attempt = std::nullopt,
        std::optional<double> energy_kj_mol = std::nullopt,
        NativeFrameEvidence evidence = std::monostate{}
    ) {
        return append_snapshot(
            snapshot_structure(session_),
            event,
            attempt,
            energy_kj_mol,
            std::move(evidence)
        );
    }

    std::optional<std::size_t> append_snapshot(
        const StructureSnapshot& snapshot,
        NativeTrajectoryEvent event,
        std::optional<std::size_t> attempt = std::nullopt,
        std::optional<double> energy_kj_mol = std::nullopt,
        NativeFrameEvidence evidence = std::monostate{}
    ) {
        if (!enabled_) {
            return std::nullopt;
        }
        const auto revision = batch_.append_topology_revision({
            snapshot.active_ligand_bond_mask,
            snapshot.active_coordination_mask,
        });
        batch_.append(NativeTrajectoryFrame{
            snapshot.coordinates,
            NativeTrajectoryStage::COORDINATION_RESTORATION,
            event,
            std::nullopt,
            attempt.has_value()
                ? std::optional<std::int32_t>(
                    static_cast<std::int32_t>(*attempt)
                )
                : std::nullopt,
            std::nullopt,
            energy_kj_mol,
            std::move(evidence),
            static_cast<std::int32_t>(revision),
        });
        return batch_.frame_count() - 1;
    }

    void select_terminal(std::optional<std::size_t> frame_index) {
        if (!frame_index.has_value()) {
            return;
        }
        batch_.select(*frame_index);
        batch_.set_terminal(*frame_index);
    }

    NativeTrajectoryBatch finish() {
        return std::move(batch_);
    }

private:
    StructureSession& session_;
    bool enabled_;
    NativeTrajectoryBatch batch_;
};


NativeCoordinationFrameEvidence coordination_evidence(
    std::optional<BondIndex> bond,
    std::optional<bool> accepted,
    std::size_t pending_count,
    bool forced,
    const detail::SegmentRingScreeningReport* screening = nullptr
) {
    return NativeCoordinationFrameEvidence{
        std::move(bond),
        accepted,
        pending_count,
        forced,
        screening == nullptr ? 0 : screening->piercing_pair_count,
        screening == nullptr ? 0 : screening->undetermined_pair_count,
        screening == nullptr ? 0 : screening->excluded_ring_count,
        std::nullopt,
        std::nullopt,
        0,
        {},
        std::nullopt,
        std::nullopt,
    };
}


NativeCoordinationFrameEvidence placement_action_evidence(
    const MetalPlacementResult& result,
    std::size_t pending_count
) {
    return NativeCoordinationFrameEvidence{
        std::nullopt,
        result.moved,
        pending_count,
        false,
        0,
        0,
        0,
        result.metal_index,
        result.moved
            ? std::optional<std::string>("relocated")
            : std::nullopt,
        result.candidates_evaluated,
        {},
        std::nullopt,
        std::nullopt,
    };
}


std::vector<std::size_t> canonical_coordination_order(
    const ComplexSessionInput& input
) {
    std::vector<std::size_t> order(input.intended_coordination_bond_count());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&input](
        std::size_t first,
        std::size_t second
    ) {
        const auto first_key = canonical_bond_key(
            input.intended_coordination_bonds[first]
        );
        const auto second_key = canonical_bond_key(
            input.intended_coordination_bonds[second]
        );
        return first_key != second_key
            ? first_key < second_key
            : first < second;
    });
    return order;
}


detail::RingWorkspaceOptions ring_workspace_options(
    const CoordinationStageOptions& options
) {
    return detail::RingWorkspaceOptions{
        detail::RingGraphScope::FULL_GRAPH,
        options.placement.maximum_actionable_ring_size,
        detail::default_maximum_relevant_cycle_count,
        options.placement.geometry_tolerances,
        options.placement.surface_limits,
    };
}


detail::SegmentRingScreeningReport screen_coordination_candidate(
    const StructureSnapshot& snapshot,
    const BondIndex& endpoints,
    const detail::PreparedRingWorkspace& workspace
) {
    return detail::screen_segment_against_rings(
        hotpot::geometry::Segment3{
            snapshot.coordinates[static_cast<std::size_t>(endpoints[0])],
            snapshot.coordinates[static_cast<std::size_t>(endpoints[1])],
        },
        workspace,
        canonical_bond_key(endpoints),
        true
    );
}


void apply_perturbation(
    StructureSession& session,
    const std::vector<Coordinate>& offsets
) {
    auto snapshot = snapshot_structure(session);
    for (std::size_t atom = 0; atom < snapshot.coordinates.size(); ++atom) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
            snapshot.coordinates[atom][axis] += offsets[atom][axis];
        }
    }
    update_structure_coordinates(session, snapshot.coordinates);
}


std::optional<double> finite_energy(double energy) {
    return std::isfinite(energy)
        ? std::optional<double>(energy)
        : std::nullopt;
}


}  // namespace


CoordinationStageResult restore_coordination(
    StructureSession& session,
    const CoordinationStageOptions& options,
    const PerturbationOffsetBatch& perturbation_offsets
) {
    options.validate();
    perturbation_offsets.validate();
    if (perturbation_offsets.atom_count != session.atom_count()) {
        throw std::invalid_argument(
            "perturbation atom_count must match the structure session"
        );
    }
    const std::size_t expected_perturbations = options.attempt_limit - 1;
    if (
        session.intended_coordination_bond_count() != 0
        && perturbation_offsets.frame_count() != expected_perturbations
    ) {
        throw std::invalid_argument(
            "coordination perturbation count must equal attempt_limit - 1"
        );
    }

    set_coordination_active_mask(
        session,
        std::vector<std::uint8_t>(
            session.intended_coordination_bond_count(),
            0
        )
    );
    CoordinationTrajectoryRecorder trajectory(
        session,
        options.trajectory_start
    );
    const auto bond_order = canonical_coordination_order(session.input());
    auto pending = bond_order;
    trajectory.append(
        NativeTrajectoryEvent::COORDINATION_READY,
        0,
        std::nullopt,
        coordination_evidence(
            std::nullopt,
            true,
            pending.size(),
            false
        )
    );

    if (pending.empty()) {
        const auto terminal = snapshot_structure(session);
        const auto terminal_index = trajectory.append(
            NativeTrajectoryEvent::TERMINAL,
            0,
            std::nullopt,
            coordination_evidence(
                std::nullopt,
                true,
                0,
                false
            )
        );
        trajectory.select_terminal(terminal_index);
        CoordinationStageResult result;
        result.selected_coordinates = terminal.coordinates;
        result.terminal_coordinates = terminal.coordinates;
        result.final_active_coordination_mask =
            terminal.active_coordination_mask;
        result.attempt_limit = options.attempt_limit;
        result.trajectory = trajectory.finish();
        result.validate();
        return result;
    }
    auto placement_snapshot = snapshot_structure(session);
    MetalPlacementReport placement_report = place_metals(
        session,
        options.placement
    );
    std::vector<std::string> warning_codes;
    for (const auto& warning : placement_report.warning_codes) {
        append_unique(warning_codes, warning);
    }
    std::size_t relocation_attempt_count = 0;
    std::vector<std::int32_t> relocated_metals;
    std::vector<std::int32_t> infeasible_metals;
    for (const auto& placement : placement_report.metals) {
        const bool relocation_attempted = placement.candidates_evaluated > 1;
        if (relocation_attempted) {
            auto trial_evidence = coordination_evidence(
                std::nullopt,
                std::nullopt,
                pending.size(),
                false
            );
            trial_evidence.metal_atom_index = placement.metal_index;
            trajectory.append_snapshot(
                placement_snapshot,
                NativeTrajectoryEvent::METAL_RELOCATION_TRIAL,
                0,
                std::nullopt,
                std::move(trial_evidence)
            );
            ++relocation_attempt_count;
        }
        if (placement.moved) {
            relocated_metals.push_back(placement.metal_index);
        }
        if (placement.status == PlacementStatus::INFEASIBLE) {
            infeasible_metals.push_back(placement.metal_index);
        }
        placement_snapshot.coordinates[
            static_cast<std::size_t>(placement.metal_index)
        ] = placement.selected_coordinates;
        if (relocation_attempted) {
            trajectory.append_snapshot(
                placement_snapshot,
                placement.moved
                    ? NativeTrajectoryEvent::METAL_RELOCATED
                    : NativeTrajectoryEvent::METAL_RELOCATION_FAILED,
                0,
                std::nullopt,
                placement_action_evidence(placement, pending.size())
            );
        }
    }

    detail::RingWorkspaceCache workspace_cache;
    const auto screening_options = ring_workspace_options(options);
    std::size_t stalled_attempts = 0;
    std::size_t rejected_piercing_trial_count = 0;
    std::size_t undetermined_trial_count = 0;
    std::size_t excluded_ring_observation_count = 0;
    std::optional<double> last_energy;

    while (!pending.empty()) {
        const auto snapshot = snapshot_structure(session);
        const auto& workspace = workspace_cache.prepare(
            session.input(),
            snapshot,
            screening_options
        );
        std::optional<std::size_t> accepted_position;
        for (std::size_t position = 0; position < pending.size(); ++position) {
            const auto bond_index = pending[position];
            const auto endpoints = session.input().intended_coordination_bonds[
                bond_index
            ];
            const auto key = canonical_bond_key(endpoints);
            trajectory.append(
                NativeTrajectoryEvent::BOND_TRIAL,
                stalled_attempts,
                std::nullopt,
                coordination_evidence(
                    key,
                    std::nullopt,
                    pending.size(),
                    false
                )
            );
            const auto screening = screen_coordination_candidate(
                snapshot,
                endpoints,
                workspace
            );
            rejected_piercing_trial_count += static_cast<std::size_t>(
                screening.piercing_pair_count != 0
            );
            undetermined_trial_count += static_cast<std::size_t>(
                screening.undetermined_pair_count != 0
            );
            excluded_ring_observation_count += screening.excluded_ring_count;
            if (screening.undetermined_pair_count != 0) {
                append_unique(
                    warning_codes,
                    "coordination_relation_undetermined"
                );
            }
            if (screening.excluded_ring_count != 0) {
                append_unique(
                    warning_codes,
                    "coordination_large_cycles_excluded"
                );
            }
            const bool accepted = screening.piercing_pair_count == 0;
            if (accepted) {
                auto active_mask = snapshot.active_coordination_mask;
                active_mask[bond_index] = 1;
                set_coordination_active_mask(session, active_mask);
                accepted_position = position;
            }
            trajectory.append(
                accepted
                    ? NativeTrajectoryEvent::BOND_ACCEPTED
                    : NativeTrajectoryEvent::BOND_REJECTED,
                stalled_attempts,
                std::nullopt,
                coordination_evidence(
                    key,
                    accepted,
                    pending.size() - static_cast<std::size_t>(accepted),
                    false,
                    &screening
                )
            );
            if (accepted) {
                break;
            }
        }

        if (accepted_position.has_value()) {
            pending.erase(
                pending.begin()
                + static_cast<std::ptrdiff_t>(*accepted_position)
            );
        } else {
            if (stalled_attempts >= options.attempt_limit) {
                break;
            }
            if (stalled_attempts != 0) {
                apply_perturbation(
                    session,
                    perturbation_offsets.frames[stalled_attempts - 1]
                );
                trajectory.append(
                    NativeTrajectoryEvent::PERTURBED,
                    stalled_attempts
                );
            }
            ++stalled_attempts;
        }

        const auto optimized = single_optimize_session(
            session,
            options.forcefield,
            options.relaxation_steps,
            options.torsion_singularity_threshold,
            options.torsion_repair_angle_radians
        );
        last_energy = finite_energy(optimized.energy_kj_mol);
        trajectory.append(
            NativeTrajectoryEvent::OPTIMIZED,
            stalled_attempts,
            last_energy
        );
    }

    std::vector<BondIndex> forced_bond_keys;
    if (!pending.empty()) {
        auto active_mask = snapshot_structure(session).active_coordination_mask;
        for (std::size_t position = 0; position < pending.size(); ++position) {
            const auto bond_index = pending[position];
            active_mask[bond_index] = 1;
            set_coordination_active_mask(session, active_mask);
            const auto key = canonical_bond_key(
                session.input().intended_coordination_bonds[bond_index]
            );
            forced_bond_keys.push_back(key);
            trajectory.append(
                NativeTrajectoryEvent::BOND_FORCED,
                stalled_attempts,
                std::nullopt,
                coordination_evidence(
                    key,
                    false,
                    pending.size() - position - 1,
                    true
                )
            );
        }
        last_energy.reset();
        append_unique(warning_codes, "coordination_bonds_forced");
    }

    const auto terminal = snapshot_structure(session);
    const bool placement_partial = std::any_of(
        placement_report.metals.begin(),
        placement_report.metals.end(),
        [](const MetalPlacementResult& result) {
            return result.status != PlacementStatus::FULLY_FEASIBLE;
        }
    );
    const auto status = forced_bond_keys.empty() && !placement_partial
        ? NativeStageStatus::COMPLETED
        : NativeStageStatus::PARTIAL;
    const auto terminal_index = trajectory.append(
        NativeTrajectoryEvent::TERMINAL,
        stalled_attempts,
        last_energy,
        coordination_evidence(
            std::nullopt,
            forced_bond_keys.empty(),
            0,
            !forced_bond_keys.empty()
        )
    );
    trajectory.select_terminal(terminal_index);

    CoordinationStageResult result;
    result.status = status;
    result.selected_coordinates = terminal.coordinates;
    result.terminal_coordinates = terminal.coordinates;
    result.final_active_coordination_mask =
        terminal.active_coordination_mask;
    result.attempt_limit = options.attempt_limit;
    result.attempts_completed = stalled_attempts;
    result.metal_relocation_attempt_count = relocation_attempt_count;
    result.relocated_metal_indices = std::move(relocated_metals);
    result.infeasible_metal_indices = std::move(infeasible_metals);
    result.forced_bond_keys = std::move(forced_bond_keys);
    result.rejected_piercing_trial_count =
        rejected_piercing_trial_count;
    result.undetermined_trial_count = undetermined_trial_count;
    result.excluded_ring_observation_count =
        excluded_ring_observation_count;
    result.warning_codes = std::move(warning_codes);
    result.trajectory = trajectory.finish();
    result.bond_count = bond_order.size();
    result.placement_report = std::move(placement_report);
    result.validate();
    return result;
}


}  // namespace hotpot::forcefields
