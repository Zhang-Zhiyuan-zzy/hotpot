#include "stage_contracts.hpp"

#include "topology_workspace.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>


namespace hotpot::forcefields {


namespace {


bool nonnegative_finite(double value) {
    return std::isfinite(value) && value >= 0.0;
}


void validate_torsion_policy(
    double singularity_threshold,
    double repair_angle_radians
) {
    if (!std::isfinite(singularity_threshold)
        || singularity_threshold < 0.0
        || singularity_threshold >= 1.0) {
        throw std::invalid_argument(
            "torsion_singularity_threshold must be finite and in [0, 1)"
        );
    }
    if (!std::isfinite(repair_angle_radians)
        || repair_angle_radians <= 0.0
        || repair_angle_radians >= 3.14159265358979323846
        || std::abs(std::sin(repair_angle_radians))
            <= singularity_threshold) {
        throw std::invalid_argument(
            "torsion_repair_angle_radians must be finite, in (0, pi), "
            "and leave the singular region"
        );
    }
}


void validate_active_mask(const std::vector<std::uint8_t>& mask) {
    if (std::any_of(mask.begin(), mask.end(), [](std::uint8_t value) {
            return value > 1;
        })) {
        throw std::invalid_argument(
            "final active coordination masks must be binary"
        );
    }
}


void validate_result_shape(
    const std::vector<Coordinate>& selected,
    const std::vector<Coordinate>& terminal,
    const std::vector<std::uint8_t>& active_mask,
    const NativeTrajectoryBatch& trajectory
) {
    if (selected.size() != terminal.size()) {
        throw std::invalid_argument(
            "selected and terminal coordinate counts must match"
        );
    }
    validate_active_mask(active_mask);
    if (trajectory.atom_count != selected.size()) {
        throw std::invalid_argument(
            "trajectory atom_count must match stage-result coordinates"
        );
    }
    if (trajectory.intended_coordination_bond_count != active_mask.size()) {
        throw std::invalid_argument(
            "trajectory coordination-bond count must match the final mask"
        );
    }
    trajectory.validate();
    if (trajectory.selected_frame_index >= 0
        && trajectory.frames[static_cast<std::size_t>(
            trajectory.selected_frame_index
        )].coordinates != selected) {
        throw std::invalid_argument(
            "selected coordinates must match the selected trajectory frame"
        );
    }
    if (trajectory.terminal_frame_index >= 0
        && trajectory.frames[static_cast<std::size_t>(
            trajectory.terminal_frame_index
        )].coordinates != terminal) {
        throw std::invalid_argument(
            "terminal coordinates must match the terminal trajectory frame"
        );
    }
}


}  // namespace


void RingScreeningOptions::validate() const {
    if (maximum_actionable_ring_size < 3) {
        throw std::invalid_argument(
            "maximum_actionable_ring_size must be at least three"
        );
    }
    if (maximum_relevant_cycle_count == 0) {
        throw std::invalid_argument(
            "maximum_relevant_cycle_count must be positive"
        );
    }
    geometry_tolerances.validate();
    surface_limits.validate();
}


void OptimizationStoppingOptions::validate() const {
    if (window == 0) {
        throw std::invalid_argument("stopping window must be positive");
    }
    if (!nonnegative_finite(maximum_energy_change_kj_mol)
        || !nonnegative_finite(maximum_atom_displacement_angstrom)
        || !nonnegative_finite(maximum_rms_gradient_kj_mol_angstrom)
        || !nonnegative_finite(maximum_gradient_kj_mol_angstrom)) {
        throw std::invalid_argument(
            "optimization stopping thresholds must be finite and "
            "nonnegative"
        );
    }
}


void CoordinationStageOptions::validate() const {
    if (forcefield.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (attempt_limit == 0) {
        throw std::invalid_argument("attempt_limit must be positive");
    }
    if (relaxation_steps == 0) {
        throw std::invalid_argument("relaxation_steps must be positive");
    }
    if (!nonnegative_finite(perturb_sigma)) {
        throw std::invalid_argument(
            "perturb_sigma must be finite and nonnegative"
        );
    }
    validate_torsion_policy(
        torsion_singularity_threshold, torsion_repair_angle_radians
    );
    placement.validate();
}


void ComplexOptimizationOptions::validate() const {
    if (forcefield.empty()) {
        throw std::invalid_argument("forcefield must not be empty");
    }
    if (algorithm != "steepest" && algorithm != "conjugate") {
        throw std::invalid_argument(
            "algorithm must be 'steepest' or 'conjugate'"
        );
    }
    if (epochs == 0) {
        throw std::invalid_argument("epochs must be positive");
    }
    if (steps_per_epoch == 0) {
        throw std::invalid_argument("steps_per_epoch must be positive");
    }
    if (untangling_attempt_limit == 0) {
        throw std::invalid_argument(
            "untangling_attempt_limit must be positive"
        );
    }
    if (perturb_interval.has_value() && *perturb_interval == 0) {
        throw std::invalid_argument(
            "perturb_interval must be positive when provided"
        );
    }
    if (!nonnegative_finite(perturb_sigma)) {
        throw std::invalid_argument(
            "perturb_sigma must be finite and nonnegative"
        );
    }
    if (!std::isfinite(vdw_cutoff_start)
        || !std::isfinite(vdw_cutoff_end)
        || vdw_cutoff_start < 0.0
        || vdw_cutoff_end < 0.0
        || (increasing_vdw && vdw_cutoff_end < vdw_cutoff_start)) {
        throw std::invalid_argument("vdw cutoffs are invalid");
    }
    if (!nonnegative_finite(energy_tolerance)) {
        throw std::invalid_argument(
            "energy_tolerance must be finite and nonnegative"
        );
    }
    if (stopping.has_value()) {
        stopping->validate();
    }
    validate_torsion_policy(
        torsion_singularity_threshold, torsion_repair_angle_radians
    );
    ring_screening.validate();
}


void NativeBondRingFinding::validate() const {
    if (ring_atom_indices.size() < 3) {
        throw std::invalid_argument(
            "bond-ring findings require at least three ring atoms"
        );
    }
    if (bond_key[0] < 0 || bond_key[1] < 0
        || bond_key[0] >= bond_key[1]) {
        throw std::invalid_argument(
            "bond-ring finding keys must be nonnegative and canonical"
        );
    }
    if (state == hotpot::geometry::PiercingState::DOES_NOT_PIERCE) {
        throw std::invalid_argument(
            "actionable findings must pierce or be undetermined"
        );
    }
    if (aabb_separated) {
        throw std::invalid_argument(
            "actionable findings cannot be AABB-separated"
        );
    }
}


void NativeRingCheckpointReport::validate() const {
    RingScreeningOptions{
        maximum_actionable_ring_size,
        maximum_relevant_cycle_count,
    }.validate();
    if (scope != NativeRingGraphScope::LIGAND_SKELETON
        && scope != NativeRingGraphScope::FULL_GRAPH) {
        throw std::invalid_argument("ring checkpoint scope is invalid");
    }
    if (selected_ring_count + excluded_ring_count != relevant_cycle_count) {
        throw std::invalid_argument(
            "selected and excluded ring counts must cover relevant cycles"
        );
    }
    if (aabb_separated_pair_count + exact_pair_count
        != candidate_pair_count) {
        throw std::invalid_argument(
            "AABB-separated and exact pair counts must cover candidates"
        );
    }
    if (piercing_pair_count + does_not_pierce_pair_count
            + undetermined_pair_count
        != candidate_pair_count) {
        throw std::invalid_argument(
            "ring relation counts must cover candidate pairs"
        );
    }
    if (actionable_findings.size()
        != piercing_pair_count + undetermined_pair_count) {
        throw std::invalid_argument(
            "actionable finding count must match actionable relations"
        );
    }
    std::size_t finding_piercing_count = 0;
    std::size_t finding_undetermined_count = 0;
    bool all_actionable_surfaces_complete = true;
    for (const auto& finding : actionable_findings) {
        finding.validate();
        all_actionable_surfaces_complete =
            all_actionable_surfaces_complete && finding.surface_complete;
        if (finding.state == hotpot::geometry::PiercingState::PIERCES) {
            ++finding_piercing_count;
        } else {
            ++finding_undetermined_count;
        }
    }
    if (finding_piercing_count != piercing_pair_count
        || finding_undetermined_count != undetermined_pair_count) {
        throw std::invalid_argument(
            "actionable finding states must match checkpoint counts"
        );
    }
    if (scan_complete && !all_actionable_surfaces_complete) {
        throw std::invalid_argument(
            "a complete scan cannot contain incomplete actionable surfaces"
        );
    }
    const auto expected_state = piercing_pair_count != 0
        ? hotpot::geometry::PiercingState::PIERCES
        : undetermined_pair_count != 0
            ? hotpot::geometry::PiercingState::UNDETERMINED
            : hotpot::geometry::PiercingState::DOES_NOT_PIERCE;
    if (state != expected_state) {
        throw std::invalid_argument(
            "ring checkpoint state must match relation counts"
        );
    }
}


detail::RingWorkspaceOptions full_graph_ring_workspace_options(
    const RingScreeningOptions& options
) {
    options.validate();
    return detail::RingWorkspaceOptions{
        detail::RingGraphScope::FULL_GRAPH,
        options.maximum_actionable_ring_size,
        options.maximum_relevant_cycle_count,
        options.geometry_tolerances,
        options.surface_limits,
    };
}


NativeRingCheckpointReport native_ring_checkpoint_report(
    const detail::BondRingCheckpoint& checkpoint
) {
    NativeRingGraphScope scope;
    switch (checkpoint.scope) {
        case detail::RingGraphScope::LIGAND_SKELETON:
            scope = NativeRingGraphScope::LIGAND_SKELETON;
            break;
        case detail::RingGraphScope::FULL_GRAPH:
            scope = NativeRingGraphScope::FULL_GRAPH;
            break;
        default:
            throw std::invalid_argument(
                "native ring checkpoint has an invalid graph scope"
            );
    }
    std::vector<NativeBondRingFinding> findings;
    findings.reserve(checkpoint.actionable_findings.size());
    for (const auto& finding : checkpoint.actionable_findings) {
        findings.push_back(NativeBondRingFinding{
            finding.ring_index,
            finding.ring_atom_indices,
            finding.bond_key,
            finding.relation.state,
            finding.relation.indeterminacy_causes,
            finding.aabb_separated,
            finding.surface_complete,
        });
    }
    NativeRingCheckpointReport report{
        checkpoint.state,
        scope,
        checkpoint.maximum_actionable_ring_size,
        checkpoint.maximum_relevant_cycle_count,
        checkpoint.relevant_cycle_count,
        checkpoint.selected_ring_count,
        checkpoint.excluded_ring_count,
        checkpoint.active_bond_count,
        checkpoint.candidate_pair_count,
        checkpoint.aabb_separated_pair_count,
        checkpoint.exact_pair_count,
        checkpoint.piercing_pair_count,
        checkpoint.does_not_pierce_pair_count,
        checkpoint.undetermined_pair_count,
        checkpoint.scan_complete,
        std::move(findings),
    };
    report.validate();
    return report;
}


std::size_t CoordinationStageResult::atom_count() const noexcept {
    return selected_coordinates.size();
}


void CoordinationStageResult::validate() const {
    validate_result_shape(
        selected_coordinates,
        terminal_coordinates,
        final_active_coordination_mask,
        trajectory
    );
    if (attempts_completed > attempt_limit) {
        throw std::invalid_argument(
            "attempts_completed must not exceed attempt_limit"
        );
    }
    if (bond_count != final_active_coordination_mask.size()) {
        throw std::invalid_argument(
            "bond_count must match the final coordination mask"
        );
    }
    if (
        !placement_report.selected_coordinates.empty()
        && placement_report.selected_coordinates.size()
            != selected_coordinates.size()
    ) {
        throw std::invalid_argument(
            "placement report atom count must match the stage result"
        );
    }
}


std::size_t ComplexOptimizationResult::atom_count() const noexcept {
    return selected_coordinates.size();
}


void ComplexOptimizationResult::validate() const {
    validate_result_shape(
        selected_coordinates,
        terminal_coordinates,
        final_active_coordination_mask,
        trajectory
    );
    if (untangling_attempts_completed > untangling_attempt_limit) {
        throw std::invalid_argument(
            "untangling attempts must not exceed the attempt limit"
        );
    }
    final_checkpoint.validate();
    if (final_checkpoint.scope != NativeRingGraphScope::FULL_GRAPH) {
        throw std::invalid_argument(
            "complex optimization requires a full-graph final checkpoint"
        );
    }
    if (std::any_of(
        final_active_coordination_mask.begin(),
        final_active_coordination_mask.end(),
        [](std::uint8_t active) { return active == 0; }
    )) {
        throw std::invalid_argument(
            "final coordination mask must be fully active"
        );
    }
    if (final_piercing_count != final_checkpoint.piercing_pair_count) {
        throw std::invalid_argument(
            "final piercing count must match the final checkpoint"
        );
    }
    if (untangling_resolved != (final_piercing_count == 0)) {
        throw std::invalid_argument(
            "untangling is resolved exactly when no piercing remains"
        );
    }
    if (selected_frame_index != trajectory.selected_frame_index) {
        throw std::invalid_argument(
            "result and trajectory selected frame indices must match"
        );
    }
    if (selected_frame_index < 0 || trajectory.terminal_frame_index < 0) {
        throw std::invalid_argument(
            "complex optimization must retain selected and terminal frames"
        );
    }
    if (minimum_piercing_count > initial_piercing_count
        || minimum_piercing_count > final_piercing_count) {
        throw std::invalid_argument(
            "minimum_piercing_count is inconsistent with stage evidence"
        );
    }
    if (energy_changes.size() != max_displacements.size()) {
        throw std::invalid_argument(
            "energy-change and displacement histories must have equal lengths"
        );
    }
    if (!epoch_energies.empty()
        && epoch_energies.size() != epochs_completed) {
        throw std::invalid_argument(
            "nonempty epoch energy history must match epochs_completed"
        );
    }
    if (termination_reason == "topology_blocked") {
        if (best_epoch != -1) {
            throw std::invalid_argument(
                "topology-blocked results cannot select an optimizer epoch"
            );
        }
        if (!std::isnan(final_energy_kj_mol)
            || !std::isnan(best_energy_kj_mol)
            || !std::isnan(rms_gradient_kj_mol_angstrom)
            || !std::isnan(max_gradient_kj_mol_angstrom)) {
            throw std::invalid_argument(
                "topology-blocked numerical metrics must be NaN"
            );
        }
        if (converged || terminal_converged) {
            throw std::invalid_argument(
                "topology-blocked results cannot be converged"
            );
        }
    }
}


std::size_t ComplexWorkflowResult::atom_count() const noexcept {
    return selected_coordinates.size();
}


void ComplexWorkflowResult::validate() const {
    coordination.validate();
    optimization.validate();
    validate_result_shape(
        selected_coordinates,
        terminal_coordinates,
        final_active_coordination_mask,
        trajectory
    );
    if (coordination.atom_count() != atom_count()
        || optimization.atom_count() != atom_count()) {
        throw std::invalid_argument(
            "composed stage results must use one stable atom order"
        );
    }
}


}  // namespace hotpot::forcefields
