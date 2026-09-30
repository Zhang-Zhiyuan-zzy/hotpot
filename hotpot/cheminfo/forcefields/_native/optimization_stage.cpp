#include "optimization_stage.hpp"

#include "untangling_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace hotpot::forcefields {


namespace detail {


bool post_checkpoint_requires_topology_blocked_tail(
    std::size_t piercing_pair_count
) noexcept {
    return piercing_pair_count != 0;
}


bool post_repair_requires_stabilization(
    std::size_t piercing_pair_count,
    bool coordinates_changed
) noexcept {
    return piercing_pair_count == 0 && coordinates_changed;
}


}  // namespace detail


namespace {


bool all_active(const std::vector<std::uint8_t>& mask) {
    return std::all_of(mask.begin(), mask.end(), [](std::uint8_t active) {
        return active != 0;
    });
}


bool finite_coordinates(const std::vector<Coordinate>& coordinates) {
    return std::all_of(
        coordinates.begin(),
        coordinates.end(),
        [](const Coordinate& coordinate) {
            return std::all_of(
                coordinate.begin(),
                coordinate.end(),
                [](double value) { return std::isfinite(value); }
            );
        }
    );
}


void append_unique(
    std::vector<std::string>& values,
    const std::string& value
) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}


std::optional<hotpot::obwrappers::StoppingCriteria> stopping_criteria(
    const std::optional<OptimizationStoppingOptions>& stopping
) {
    if (!stopping.has_value()) {
        return std::nullopt;
    }
    return hotpot::obwrappers::StoppingCriteria{
        stopping->window,
        stopping->maximum_energy_change_kj_mol,
        stopping->maximum_atom_displacement_angstrom,
        stopping->maximum_rms_gradient_kj_mol_angstrom,
        stopping->maximum_gradient_kj_mol_angstrom,
    };
}


hotpot::obwrappers::OptimizationOptions optimizer_options(
    const ComplexOptimizationOptions& options
) {
    return hotpot::obwrappers::OptimizationOptions{
        options.forcefield,
        options.algorithm,
        options.epochs,
        options.steps_per_epoch,
        options.perturb_interval,
        options.frame_detail != FrameDetail::NONE,
        options.retain_epoch_history,
        options.increasing_vdw,
        options.vdw_cutoff_start,
        options.vdw_cutoff_end,
        options.energy_tolerance,
        stopping_criteria(options.stopping),
    };
}


hotpot::obwrappers::OptimizationOptions stabilization_options(
    const ComplexOptimizationOptions& options
) {
    auto stabilization = optimizer_options(options);
    stabilization.epochs = 1;
    stabilization.perturb_interval = std::nullopt;
    return stabilization;
}


RingUntanglingOptions untangling_options(
    const ComplexOptimizationOptions& options,
    const detail::RingWorkspaceOptions& ring_workspace
) {
    return RingUntanglingOptions{
        options.forcefield,
        options.steps_per_epoch,
        options.torsion_singularity_threshold,
        options.torsion_repair_angle_radians,
        ring_workspace,
    };
}


detail::BondRingCheckpoint full_checkpoint(
    StructureSession& session,
    const detail::RingWorkspaceOptions& options,
    detail::RingWorkspaceCache& cache
) {
    const auto snapshot = snapshot_structure(session);
    return detail::scan_bond_ring_checkpoint(
        cache.prepare(session.input(), snapshot, options),
        options
    );
}


void validate_perturbation_batch(
    const PerturbationOffsetBatch& batch,
    std::size_t atom_count,
    std::size_t expected_frame_count,
    const char* label
) {
    batch.validate();
    if (batch.atom_count != atom_count) {
        throw std::invalid_argument(
            std::string(label) + " atom_count must match the session"
        );
    }
    if (batch.frame_count() != expected_frame_count) {
        throw std::invalid_argument(
            std::string(label) + " frame count does not match its schedule"
        );
    }
}


std::size_t expected_optimization_offset_count(
    const ComplexOptimizationOptions& options
) {
    return options.perturb_interval.has_value()
        ? (options.epochs - 1) / *options.perturb_interval
        : 0;
}


hotpot::obwrappers::OptimizationResult topology_blocked_result(
    const StructureSnapshot& snapshot
) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return hotpot::obwrappers::OptimizationResult{
        snapshot.coordinates,
        snapshot.coordinates,
        {},
        -1,
        -1,
        nan,
        nan,
        nan,
        nan,
        false,
        false,
        0,
        0,
        0,
        0,
        {},
        "topology_blocked",
        false,
        {},
        {},
        {},
        {hotpot::obwrappers::RuleStage::PRE_FORCEFIELD_SETUP, {}},
    };
}


void append_untangling_warnings(
    std::vector<std::string>& warning_codes,
    const RingUntanglingResult& untangling
) {
    for (const auto& warning : untangling.warning_codes) {
        append_unique(warning_codes, warning);
    }
}


NativeRingFrameEvidence checkpoint_evidence(
    const detail::BondRingCheckpoint& checkpoint
) {
    return NativeRingFrameEvidence{
        checkpoint.piercing_pair_count,
        checkpoint.undetermined_pair_count,
        checkpoint.scope == detail::RingGraphScope::FULL_GRAPH
            ? std::optional<std::string>("full_graph")
            : std::optional<std::string>("ligand_skeleton"),
        checkpoint.maximum_actionable_ring_size,
        checkpoint.selected_ring_count,
        checkpoint.excluded_ring_count,
        checkpoint.candidate_pair_count,
        checkpoint.aabb_separated_pair_count,
        checkpoint.exact_pair_count,
        checkpoint.does_not_pierce_pair_count,
        checkpoint.scan_complete,
    };
}


NativeOptimizationFrameEvidence optimization_evidence(
    const hotpot::obwrappers::OptimizationFrame& frame
) {
    return NativeOptimizationFrameEvidence{
        frame.converged,
        frame.exploded,
        finite_coordinates(frame.coordinates),
        std::isfinite(frame.energy),
        std::isfinite(frame.rms_gradient)
            && std::isfinite(frame.max_gradient),
        std::isfinite(frame.rms_gradient)
            ? std::optional<double>(frame.rms_gradient)
            : std::nullopt,
        std::isfinite(frame.max_gradient)
            ? std::optional<double>(frame.max_gradient)
            : std::nullopt,
        frame.energy_change,
        frame.max_displacement,
    };
}


NativeOptimizationFrameEvidence optimization_evidence(
    const hotpot::obwrappers::OptimizationResult& result
) {
    return NativeOptimizationFrameEvidence{
        result.converged,
        result.exploded,
        finite_coordinates(result.coordinates),
        std::isfinite(result.best_energy),
        std::isfinite(result.rms_gradient)
            && std::isfinite(result.max_gradient),
        std::isfinite(result.rms_gradient)
            ? std::optional<double>(result.rms_gradient)
            : std::nullopt,
        std::isfinite(result.max_gradient)
            ? std::optional<double>(result.max_gradient)
            : std::nullopt,
        std::nullopt,
        std::nullopt,
    };
}


NativeTrajectoryEvent trajectory_event(RingUntanglingEvent event) {
    switch (event) {
        case RingUntanglingEvent::TOPOLOGY_CHECKPOINT:
            return NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT;
        case RingUntanglingEvent::RING_OPENED:
            return NativeTrajectoryEvent::RING_OPENED;
        case RingUntanglingEvent::PERTURBED:
            return NativeTrajectoryEvent::PERTURBED;
        case RingUntanglingEvent::OPEN_TOPOLOGY_OPTIMIZED:
            return NativeTrajectoryEvent::OPTIMIZED;
        case RingUntanglingEvent::RING_CLOSED:
            return NativeTrajectoryEvent::RING_CLOSED;
        case RingUntanglingEvent::ROLLED_BACK:
            return NativeTrajectoryEvent::ROLLED_BACK;
    }
    throw std::invalid_argument("unknown ring-untangling event");
}


class OptimizationStageTrajectory final {
public:
    OptimizationStageTrajectory(
        StructureSession& session,
        NativeTrajectoryStart start,
        FrameDetail frame_detail
    ) :
        frame_detail_(frame_detail),
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

    std::optional<std::size_t> append_checkpoint(
        const StructureSnapshot& snapshot,
        const detail::BondRingCheckpoint& checkpoint,
        NativeTrajectoryStage stage,
        std::optional<std::size_t> attempt = std::nullopt,
        std::optional<double> energy = std::nullopt
    ) {
        if (!records(stage)) {
            return std::nullopt;
        }
        return append_snapshot(
            snapshot,
            stage,
            NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT,
            attempt,
            std::nullopt,
            energy,
            checkpoint_evidence(checkpoint)
        );
    }

    std::optional<std::size_t> append_untangling(
        const RingUntanglingResult& untangling,
        bool skip_entry_checkpoint
    ) {
        if (!records(NativeTrajectoryStage::COMPLEX_UNTANGLING)) {
            return std::nullopt;
        }
        std::optional<std::size_t> last_index;
        for (std::size_t index = 0; index < untangling.steps.size(); ++index) {
            const auto& step = untangling.steps[index];
            if (skip_entry_checkpoint && index == 0
                && step.event
                    == RingUntanglingEvent::TOPOLOGY_CHECKPOINT) {
                continue;
            }
            NativeFrameEvidence evidence = std::monostate{};
            if (step.checkpoint_evidence.has_value()) {
                evidence = checkpoint_evidence(*step.checkpoint_evidence);
            } else if (step.observed_state.has_value()) {
                evidence = NativeRingFrameEvidence{
                    step.confirmed_piercing_count.value_or(0),
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                    std::nullopt,
                };
            }
            last_index = append_snapshot(
                step.snapshot,
                NativeTrajectoryStage::COMPLEX_UNTANGLING,
                trajectory_event(step.event),
                step.attempt,
                std::nullopt,
                step.energy_kj_mol,
                std::move(evidence)
            );
        }
        return last_index;
    }

    std::optional<std::size_t> append_optimizer(
        const StructureSnapshot& input_snapshot,
        const hotpot::obwrappers::OptimizationResult& optimization,
        NativeTrajectoryStage stage,
        std::size_t attempt
    ) {
        if (!records(stage)) {
            return std::nullopt;
        }
        append_snapshot(
            input_snapshot,
            stage,
            NativeTrajectoryEvent::INITIAL,
            attempt
        );
        std::vector<std::size_t> epoch_frame_indices;
        if (frame_detail_ != FrameDetail::NONE) {
            epoch_frame_indices.reserve(optimization.frames.size());
            for (const auto& frame : optimization.frames) {
                epoch_frame_indices.push_back(append_state(
                    frame.coordinates,
                    input_snapshot,
                    stage,
                    NativeTrajectoryEvent::EPOCH_COMPLETE,
                    attempt,
                    frame.epoch_index,
                    std::isfinite(frame.energy)
                        ? std::optional<double>(frame.energy)
                        : std::nullopt,
                    optimization_evidence(frame)
                ));
            }
        }
        if (optimization.selected_frame_index >= 0
            && static_cast<std::size_t>(optimization.selected_frame_index)
                < epoch_frame_indices.size()) {
            return epoch_frame_indices[static_cast<std::size_t>(
                optimization.selected_frame_index
            )];
        }
        return append_state(
            optimization.coordinates,
            input_snapshot,
            stage,
            NativeTrajectoryEvent::OPTIMIZED,
            attempt,
            std::nullopt,
            std::isfinite(optimization.best_energy)
                ? std::optional<double>(optimization.best_energy)
                : std::nullopt,
            optimization_evidence(optimization)
        );
    }

    std::size_t append_stabilization(
        const StructureSnapshot& input_snapshot,
        const hotpot::obwrappers::OptimizationResult& optimization,
        std::size_t attempt
    ) {
        const auto stage = records(
            NativeTrajectoryStage::COMPLEX_UNTANGLING
        )
            ? NativeTrajectoryStage::COMPLEX_UNTANGLING
            : NativeTrajectoryStage::FINAL_OPTIMIZATION;
        std::vector<std::size_t> epoch_frame_indices;
        if (frame_detail_ != FrameDetail::NONE) {
            epoch_frame_indices.reserve(optimization.frames.size());
            for (const auto& frame : optimization.frames) {
                epoch_frame_indices.push_back(append_state(
                    frame.coordinates,
                    input_snapshot,
                    stage,
                    NativeTrajectoryEvent::EPOCH_COMPLETE,
                    attempt,
                    frame.epoch_index,
                    std::isfinite(frame.energy)
                        ? std::optional<double>(frame.energy)
                        : std::nullopt,
                    optimization_evidence(frame)
                ));
            }
        }
        const auto settled_index = append_state(
            optimization.coordinates,
            input_snapshot,
            stage,
            NativeTrajectoryEvent::SETTLED,
            attempt,
            std::nullopt,
            std::isfinite(optimization.best_energy)
                ? std::optional<double>(optimization.best_energy)
                : std::nullopt,
            optimization_evidence(optimization)
        );
        if (optimization.selected_frame_index >= 0
            && static_cast<std::size_t>(optimization.selected_frame_index)
                < epoch_frame_indices.size()) {
            return epoch_frame_indices[static_cast<std::size_t>(
                optimization.selected_frame_index
            )];
        }
        return settled_index;
    }

    std::size_t ensure_final_selection(
        std::optional<std::size_t> selected_index,
        const StructureSnapshot& selected_snapshot,
        const detail::BondRingCheckpoint& checkpoint,
        const hotpot::obwrappers::OptimizationResult& optimization
    ) {
        if (selected_index.has_value()
            && batch_.frames[*selected_index].coordinates
                == selected_snapshot.coordinates) {
            return *selected_index;
        }
        const bool topology_blocked =
            optimization.termination_reason == "topology_blocked";
        return append_snapshot(
            selected_snapshot,
            NativeTrajectoryStage::FINAL_OPTIMIZATION,
            topology_blocked
                ? NativeTrajectoryEvent::TOPOLOGY_CHECKPOINT
                : NativeTrajectoryEvent::OPTIMIZED,
            std::nullopt,
            std::nullopt,
            std::isfinite(optimization.best_energy)
                ? std::optional<double>(optimization.best_energy)
                : std::nullopt,
            topology_blocked
                ? NativeFrameEvidence(checkpoint_evidence(checkpoint))
                : NativeFrameEvidence(optimization_evidence(optimization))
        );
    }

    std::size_t append_terminal(
        const std::vector<Coordinate>& coordinates,
        const StructureSnapshot& topology,
        const hotpot::obwrappers::OptimizationResult& optimization
    ) {
        return append_state(
            coordinates,
            topology,
            NativeTrajectoryStage::FINAL_OPTIMIZATION,
            NativeTrajectoryEvent::TERMINAL,
            std::nullopt,
            std::nullopt,
            std::isfinite(optimization.final_energy)
                ? std::optional<double>(optimization.final_energy)
                : std::nullopt,
            std::monostate{}
        );
    }

    NativeTrajectoryBatch finish(
        std::size_t selected_index,
        std::size_t terminal_index
    ) {
        batch_.select(selected_index);
        batch_.set_terminal(terminal_index);
        return std::move(batch_);
    }

private:
    bool records(NativeTrajectoryStage stage) const noexcept {
        return static_cast<std::int32_t>(batch_.start)
            <= static_cast<std::int32_t>(stage);
    }

    std::size_t append_snapshot(
        const StructureSnapshot& snapshot,
        NativeTrajectoryStage stage,
        NativeTrajectoryEvent event,
        std::optional<std::size_t> attempt = std::nullopt,
        std::optional<std::size_t> step = std::nullopt,
        std::optional<double> energy = std::nullopt,
        NativeFrameEvidence evidence = std::monostate{}
    ) {
        return append_state(
            snapshot.coordinates,
            snapshot,
            stage,
            event,
            attempt,
            step,
            energy,
            std::move(evidence)
        );
    }

    std::size_t append_state(
        const std::vector<Coordinate>& coordinates,
        const StructureSnapshot& topology,
        NativeTrajectoryStage stage,
        NativeTrajectoryEvent event,
        std::optional<std::size_t> attempt = std::nullopt,
        std::optional<std::size_t> step = std::nullopt,
        std::optional<double> energy = std::nullopt,
        NativeFrameEvidence evidence = std::monostate{}
    ) {
        const auto revision = batch_.append_topology_revision({
            topology.active_ligand_bond_mask,
            topology.active_coordination_mask,
        });
        batch_.append(NativeTrajectoryFrame{
            coordinates,
            stage,
            event,
            std::nullopt,
            attempt.has_value()
                ? std::optional<std::int32_t>(
                    static_cast<std::int32_t>(*attempt)
                )
                : std::nullopt,
            step.has_value()
                ? std::optional<std::int64_t>(
                    static_cast<std::int64_t>(*step)
                )
                : std::nullopt,
            energy,
            std::move(evidence),
            static_cast<std::int32_t>(revision),
        });
        return batch_.frame_count() - 1;
    }

    FrameDetail frame_detail_;
    NativeTrajectoryBatch batch_;
};


}  // namespace


namespace {


hotpot::obwrappers::OptimizationResult combine_optimization_segments(
    const std::vector<hotpot::obwrappers::OptimizationResult>& segments
) {
    if (segments.empty()) {
        throw std::invalid_argument(
            "at least one optimization segment is required"
        );
    }
    auto combined = segments.front();
    if (combined.frames.empty()) {
        combined.selected_frame_index = -1;
    }
    std::size_t preceding_epochs = combined.epochs_completed;
    std::size_t preceding_frames = combined.frames.size();
    std::size_t frame_segment_offset = 0;
    for (const auto& frame : combined.frames) {
        frame_segment_offset = std::max(
            frame_segment_offset, frame.segment_index + 1
        );
    }

    for (std::size_t index = 1; index < segments.size(); ++index) {
        const auto& segment = segments[index];
        for (auto frame : segment.frames) {
            frame.epoch_index += preceding_epochs;
            frame.segment_index += frame_segment_offset;
            combined.frames.push_back(std::move(frame));
        }
        std::size_t local_segment_count = 0;
        for (const auto& frame : segment.frames) {
            local_segment_count = std::max(
                local_segment_count, frame.segment_index + 1
            );
        }
        frame_segment_offset += local_segment_count;

        combined.coordinates = segment.coordinates;
        combined.terminal_coordinates = segment.terminal_coordinates;
        combined.selected_frame_index = segment.selected_frame_index < 0
                || segment.frames.empty()
            ? -1
            : static_cast<long>(preceding_frames)
                + segment.selected_frame_index;
        combined.best_epoch = segment.best_epoch < 0
            ? -1
            : static_cast<long>(preceding_epochs) + segment.best_epoch;
        combined.final_energy = segment.final_energy;
        combined.best_energy = segment.best_energy;
        combined.rms_gradient = segment.rms_gradient;
        combined.max_gradient = segment.max_gradient;
        combined.exploded = segment.exploded;
        combined.converged = segment.converged;
        combined.terminal_converged = segment.terminal_converged;
        combined.selected_segment_epochs_completed =
            segment.selected_segment_epochs_completed;
        combined.backend_energy_unit = segment.backend_energy_unit;
        combined.termination_reason = segment.termination_reason;
        combined.energy_changes = segment.energy_changes;
        combined.max_displacements = segment.max_displacements;
        combined.epochs_completed += segment.epochs_completed;
        combined.steps_submitted += segment.steps_submitted;
        combined.initialization_steps += segment.initialization_steps;
        combined.epoch_energies.insert(
            combined.epoch_energies.end(),
            segment.epoch_energies.begin(),
            segment.epoch_energies.end()
        );
        combined.rules.applications.insert(
            combined.rules.applications.end(),
            segment.rules.applications.begin(),
            segment.rules.applications.end()
        );
        preceding_epochs += segment.epochs_completed;
        preceding_frames += segment.frames.size();
    }
    return combined;
}


}  // namespace


ComplexOptimizationResult optimize_complex(
    StructureSession& session,
    const ComplexOptimizationOptions& options,
    const PerturbationOffsetBatch& untangling_offsets,
    const PerturbationOffsetBatch& optimization_offsets
) {
    options.validate();
    const auto maximum_steps = static_cast<std::size_t>(
        std::numeric_limits<int>::max()
    );
    if (options.epochs > maximum_steps / options.steps_per_epoch) {
        throw std::invalid_argument(
            "epochs * steps_per_epoch exceeds the Open Babel step limit"
        );
    }
    validate_perturbation_batch(
        untangling_offsets,
        session.atom_count(),
        options.untangling_attempt_limit,
        "untangling perturbation"
    );
    validate_perturbation_batch(
        optimization_offsets,
        session.atom_count(),
        expected_optimization_offset_count(options),
        "optimization perturbation"
    );
    const auto entry_snapshot = snapshot_structure(session);
    if (!all_active(entry_snapshot.active_ligand_bond_mask)
        || !all_active(entry_snapshot.active_coordination_mask)) {
        throw std::invalid_argument(
            "complex optimization requires a fully assembled topology"
        );
    }

    OptimizationStageTrajectory trajectory(
        session,
        options.trajectory_start,
        options.frame_detail
    );
    const auto ring_workspace = full_graph_ring_workspace_options(
        options.ring_screening
    );
    auto ring_options = untangling_options(options, ring_workspace);
    RingUntanglingAttemptCursor attempt_cursor{
        options.untangling_attempt_limit,
        0,
    };
    std::optional<std::size_t> selected_frame_candidate;
    detail::RingWorkspaceCache checkpoint_cache;
    auto checkpoint = full_checkpoint(
        session, ring_workspace, checkpoint_cache
    );
    selected_frame_candidate = trajectory.append_checkpoint(
        entry_snapshot,
        checkpoint,
        NativeTrajectoryStage::COMPLEX_UNTANGLING,
        0
    );
    const std::size_t initial_piercing_count =
        checkpoint.piercing_pair_count;
    std::size_t minimum_piercing_count = initial_piercing_count;
    std::vector<std::string> warning_codes;
    std::vector<hotpot::obwrappers::OptimizationResult> segments;
    std::optional<std::vector<Coordinate>> numerical_coordinates;

    if (checkpoint.piercing_pair_count != 0) {
        const auto untangling = untangle_ring_piercings(
            session,
            ring_options,
            untangling_offsets,
            attempt_cursor,
            checkpoint
        );
        checkpoint = untangling.final_checkpoint;
        if (const auto frame_index = trajectory.append_untangling(
                untangling, true
            )) {
            selected_frame_candidate = frame_index;
        }
        minimum_piercing_count = std::min(
            minimum_piercing_count, untangling.minimum_piercing_count
        );
        append_untangling_warnings(warning_codes, untangling);
    }

    if (checkpoint.piercing_pair_count != 0) {
        append_unique(
            warning_codes, "ring_piercing_blocks_complex_optimization"
        );
        segments.push_back(topology_blocked_result(
            snapshot_structure(session)
        ));
    } else {
        const auto optimizer_input = snapshot_structure(session);
        segments.push_back(optimize_session(
            session,
            optimizer_options(options),
            optimization_offsets,
            options.torsion_singularity_threshold,
            options.torsion_repair_angle_radians
        ));
        selected_frame_candidate = trajectory.append_optimizer(
            optimizer_input,
            segments.back(),
            NativeTrajectoryStage::FINAL_OPTIMIZATION,
            0
        );
        numerical_coordinates = snapshot_structure(session).coordinates;
        checkpoint = full_checkpoint(
            session, ring_workspace, checkpoint_cache
        );
        const auto post_optimizer_checkpoint_index =
            trajectory.append_checkpoint(
            snapshot_structure(session),
            checkpoint,
            NativeTrajectoryStage::FINAL_OPTIMIZATION,
            0,
            std::isfinite(segments.back().best_energy)
                ? std::optional<double>(segments.back().best_energy)
                : std::nullopt
        );
        if (checkpoint.piercing_pair_count != 0
            && post_optimizer_checkpoint_index.has_value()) {
            selected_frame_candidate = post_optimizer_checkpoint_index;
        }

        while (checkpoint.piercing_pair_count != 0
            && !attempt_cursor.exhausted()) {
            const auto untangling = untangle_ring_piercings(
                session,
                ring_options,
                untangling_offsets,
                attempt_cursor,
                checkpoint
            );
            checkpoint = untangling.final_checkpoint;
            if (const auto frame_index = trajectory.append_untangling(
                    untangling, true
                )) {
                selected_frame_candidate = frame_index;
            }
            minimum_piercing_count = std::min(
                minimum_piercing_count, untangling.minimum_piercing_count
            );
            append_untangling_warnings(warning_codes, untangling);
            if (untangling.attempts_used == 0) {
                break;
            }
            if (checkpoint.piercing_pair_count != 0) {
                continue;
            }
            const auto repaired_coordinates = snapshot_structure(
                session
            ).coordinates;
            if (!detail::post_repair_requires_stabilization(
                    checkpoint.piercing_pair_count,
                    repaired_coordinates != *numerical_coordinates
                )) {
                continue;
            }

            const PerturbationOffsetBatch no_offsets{
                session.atom_count(), {}
            };
            const auto stabilization_input = snapshot_structure(session);
            segments.push_back(optimize_session(
                session,
                stabilization_options(options),
                no_offsets,
                options.torsion_singularity_threshold,
                options.torsion_repair_angle_radians
            ));
            selected_frame_candidate = trajectory.append_stabilization(
                stabilization_input,
                segments.back(),
                segments.size() - 1
            );
            numerical_coordinates = snapshot_structure(session).coordinates;
            checkpoint = full_checkpoint(
                session, ring_workspace, checkpoint_cache
            );
            const auto post_stabilization_checkpoint_index =
                trajectory.append_checkpoint(
                snapshot_structure(session),
                checkpoint,
                NativeTrajectoryStage::COMPLEX_UNTANGLING,
                attempt_cursor.attempts_completed,
                std::isfinite(segments.back().best_energy)
                    ? std::optional<double>(segments.back().best_energy)
                    : std::nullopt
            );
            if (checkpoint.piercing_pair_count != 0
                && post_stabilization_checkpoint_index.has_value()) {
                selected_frame_candidate =
                    post_stabilization_checkpoint_index;
            }
        }

        if (detail::post_checkpoint_requires_topology_blocked_tail(
                checkpoint.piercing_pair_count
            )) {
            segments.push_back(topology_blocked_result(
                snapshot_structure(session)
            ));
        }
    }

    if (checkpoint.state == hotpot::geometry::PiercingState::UNDETERMINED) {
        append_unique(warning_codes, "ring_relation_undetermined");
    }
    const auto optimization = combine_optimization_segments(segments);
    const auto final_snapshot = snapshot_structure(session);
    const bool resolved = checkpoint.piercing_pair_count == 0;
    minimum_piercing_count = std::min(
        minimum_piercing_count, checkpoint.piercing_pair_count
    );
    const auto selected_frame_index = trajectory.ensure_final_selection(
        selected_frame_candidate, final_snapshot, checkpoint, optimization
    );
    const auto terminal_frame_index = trajectory.append_terminal(
        optimization.terminal_coordinates,
        final_snapshot,
        optimization
    );
    auto trajectory_batch = trajectory.finish(
        selected_frame_index, terminal_frame_index
    );
    ComplexOptimizationResult result{
        resolved ? NativeStageStatus::COMPLETED : NativeStageStatus::PARTIAL,
        final_snapshot.coordinates,
        optimization.terminal_coordinates,
        final_snapshot.active_coordination_mask,
        attempt_cursor.attempt_limit,
        attempt_cursor.attempts_completed,
        initial_piercing_count,
        checkpoint.piercing_pair_count,
        minimum_piercing_count,
        resolved,
        static_cast<std::int64_t>(selected_frame_index),
        optimization.best_epoch,
        optimization.final_energy,
        optimization.best_energy,
        optimization.rms_gradient,
        optimization.max_gradient,
        optimization.energy_changes,
        optimization.max_displacements,
        optimization.epoch_energies,
        optimization.exploded,
        optimization.converged,
        optimization.terminal_converged,
        optimization.epochs_completed,
        optimization.steps_submitted,
        optimization.initialization_steps,
        optimization.selected_segment_epochs_completed,
        optimization.backend_energy_unit,
        optimization.termination_reason,
        warning_codes,
        std::move(trajectory_batch),
        native_ring_checkpoint_report(checkpoint),
    };
    result.validate();
    return result;
}


}  // namespace hotpot::forcefields
